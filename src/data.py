import pickle
from pathlib import Path

import numpy as np
import torch
import torch_geometric.transforms as T
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Coauthor, Planetoid
from torch_geometric.transforms import GDC
from torch_geometric.utils import (
    add_self_loops,
    add_remaining_self_loops,
    degree,
    dense_to_sparse,
    remove_self_loops,
    subgraph,
    to_dense_adj,
    to_undirected,
)
from torch_sparse import coalesce

from heterophilic import Actor, WebKB


SUPPORTED_DATASETS = (
    "Cora",
    "Citeseer",
    "Pubmed",
    "CoauthorCS",
    "Computers",
    "Photo",
    "texas",
    "wisconsin",
    "cornell",
    "film",
)

HOMOPHILIC_DATASETS = {
    "Cora", "Citeseer", "Pubmed", "CoauthorCS", "Computers", "Photo",
}
PLANETOID_DATASETS = {"Cora", "Citeseer", "Pubmed"}


class _GDCPositionEncoding(GDC):
    """Expose the diffusion matrix used by the optional position encoder."""

    def position_encoding(self, data):
        num_nodes = int(data.num_nodes)
        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_attr", None)
        if edge_weight is None:
            edge_weight = torch.ones(
                edge_index.size(1),
                device=edge_index.device,
                dtype=torch.float32,
            )
        elif edge_weight.ndim != 1:
            raise ValueError("GDC position encoding requires scalar edge weights.")

        if self.self_loop_weight:
            edge_index, edge_weight = add_self_loops(
                edge_index,
                edge_weight,
                fill_value=float(self.self_loop_weight),
                num_nodes=num_nodes,
            )
        edge_index, edge_weight = coalesce(
            edge_index,
            edge_weight,
            num_nodes,
            num_nodes,
        )

        if self.exact:
            edge_index, edge_weight = self.transition_matrix(
                edge_index,
                edge_weight,
                num_nodes,
                self.normalization_in,
            )
            diffusion = self.diffusion_matrix_exact(
                edge_index,
                edge_weight,
                num_nodes,
                **self.diffusion_kwargs,
            )
            edge_index, edge_weight = dense_to_sparse(diffusion)
        else:
            edge_index, edge_weight = self.diffusion_matrix_approx(
                edge_index,
                edge_weight,
                num_nodes,
                self.normalization_in,
                **self.diffusion_kwargs,
            )

        edge_index, edge_weight = coalesce(
            edge_index,
            edge_weight,
            num_nodes,
            num_nodes,
        )
        edge_index, edge_weight = self.transition_matrix(
            edge_index,
            edge_weight,
            num_nodes,
            self.normalization_out,
        )
        return to_dense_adj(
            edge_index,
            edge_attr=edge_weight,
            max_num_nodes=num_nodes,
        ).squeeze(0)


def _as_position_tensor(value):
    if isinstance(value, dict) and "data" in value:
        value = value["data"]
    if hasattr(value, "toarray"):
        value = value.toarray()
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.detach().cpu().to(dtype=torch.float32)


def load_position_encoding(data, opt, data_dir):
    """Load a cached encoding or construct the configured dense GDC matrix."""

    encoding_type = str(opt["pos_enc_type"])
    cache_dir = Path(data_dir) / "pos_encodings"
    if encoding_type == "GDC":
        alpha = format(float(opt["ppr_alpha"]), ".8g").replace(".", "p")
        epsilon = format(float(opt["position_eps"]), ".8g").replace(".", "p")
        solver = "exact" if opt["exact"] else "approx"
        cache_name = (
            f"{opt['dataset']}_GDC_{opt['pos_enc_orientation']}_"
            f"a{alpha}_eps{epsilon}_{solver}.pkl"
        )
    else:
        cache_name = f"{opt['dataset']}_{encoding_type}.pkl"
    cache_path = cache_dir / cache_name

    if cache_path.exists():
        with cache_path.open("rb") as handle:
            encoding = _as_position_tensor(pickle.load(handle))
    elif encoding_type == "GDC":
        transform = _GDCPositionEncoding(
            self_loop_weight=1.0,
            normalization_in="sym",
            normalization_out="col",
            diffusion_kwargs={
                "method": "ppr",
                "alpha": float(opt["ppr_alpha"]),
                "eps": float(opt["position_eps"]),
            },
            sparsification_kwargs={
                "method": "threshold",
                "eps": float(opt["position_eps"]),
            },
            exact=bool(opt["exact"]),
        )
        encoding = _as_position_tensor(transform.position_encoding(data))
        if opt["pos_enc_orientation"] == "col":
            encoding = encoding.t().contiguous()
        cache_dir.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump(encoding, handle)
    else:
        raise FileNotFoundError(
            f"Position encoding {encoding_type!r} requires the original cache "
            f"file: {cache_path}"
        )

    if encoding.ndim != 2 or encoding.shape[0] != int(data.num_nodes):
        raise ValueError(
            "Position encoding must have shape [num_nodes, dim], "
            f"received {tuple(encoding.shape)} for {int(data.num_nodes)} nodes."
        )
    if not torch.isfinite(encoding).all():
        raise ValueError("Position encoding contains non-finite values.")
    return encoding


def _load_raw_dataset(name: str, data_dir: Path):
    root = str(data_dir / name)
    if name in {"Cora", "Citeseer", "Pubmed"}:
        return Planetoid(root, name)
    if name in {"Computers", "Photo"}:
        return Amazon(root, name)
    if name == "CoauthorCS":
        return Coauthor(root, "CS")
    if name in {"cornell", "texas", "wisconsin"}:
        return WebKB(root=root, name=name, transform=T.NormalizeFeatures())
    if name == "film":
        return Actor(root=root, transform=T.NormalizeFeatures())
    raise ValueError(f"Unsupported main-experiment dataset: {name}")


def _largest_component_nodes(data: Data) -> torch.Tensor:
    edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    row = edge_index[0].detach().cpu().numpy()
    col = edge_index[1].detach().cpu().numpy()
    adjacency = coo_matrix(
        (np.ones(row.shape[0], dtype=np.float64), (row, col)),
        shape=(data.num_nodes, data.num_nodes),
    ).tocsr()
    component_count, labels = connected_components(adjacency, directed=False)
    if component_count == 1:
        return torch.arange(data.num_nodes, dtype=torch.long)
    largest_label = int(np.argmax(np.bincount(labels)))
    return torch.from_numpy(np.flatnonzero(labels == largest_label)).long()


def _induced_data(data: Data, nodes: torch.Tensor) -> Data:
    if nodes.numel() == data.num_nodes:
        return data

    edge_index, _ = subgraph(
        nodes,
        data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
    )
    induced = Data(x=data.x[nodes], y=data.y[nodes], edge_index=edge_index)
    for name in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(data, name, None)
        if mask is not None:
            setattr(induced, name, mask[nodes])
    return induced


def _development_split(
    data: Data,
    seed: int,
    *,
    development_size: int,
    train_per_class: int = 20,
) -> None:
    """Use the stratified ACSG/GRAND main-experiment split."""
    generator = np.random.RandomState(seed)
    labels = data.y.view(-1).detach().cpu().numpy()
    development_size = min(int(development_size), int(data.num_nodes))
    development = generator.choice(
        data.num_nodes,
        size=development_size,
        replace=False,
    )
    development_set = set(development.tolist())
    test_indices = np.asarray(
        [index for index in range(data.num_nodes) if index not in development_set],
        dtype=np.int64,
    )

    train_indices = []
    generator = np.random.RandomState(seed)
    for label in np.unique(labels):
        candidates = development[labels[development] == label]
        count = min(train_per_class, candidates.size)
        selected = generator.choice(candidates, size=count, replace=False)
        train_indices.extend(selected.tolist())
    train_set = set(train_indices)
    validation_indices = np.asarray(
        [index for index in development if int(index) not in train_set],
        dtype=np.int64,
    )

    def mask(indices):
        value = torch.zeros(data.num_nodes, dtype=torch.bool)
        value[torch.as_tensor(indices, dtype=torch.long)] = True
        return value

    data.train_mask = mask(train_indices)
    data.val_mask = mask(validation_indices)
    data.test_mask = mask(test_indices)


def load_dataset(
    name: str,
    data_dir: str,
    *,
    split_index: int,
    split_seed: int,
):
    """Load one paper dataset, retain its LCC, and select one split."""
    dataset = _load_raw_dataset(name, Path(data_dir))
    data = _induced_data(dataset.data, _largest_component_nodes(dataset.data))

    if name in HOMOPHILIC_DATASETS:
        if name in PLANETOID_DATASETS:
            split_seed = int(np.random.RandomState(split_seed).randint(0, 1000))
        else:
            split_seed = 12345
        _development_split(
            data,
            split_seed,
            development_size=5000 if name == "CoauthorCS" else 1500,
        )
    elif getattr(data, "train_mask", None) is None:
        _development_split(data, split_seed, development_size=1500)
    elif data.train_mask.ndim == 2:
        split_count = int(data.train_mask.shape[1])
        if not 0 <= split_index < split_count:
            raise ValueError(
                f"split_index must be in [0, {split_count - 1}] for {name}."
            )
        data.train_mask = data.train_mask[:, split_index]
        data.val_mask = data.val_mask[:, split_index]
        data.test_mask = data.test_mask[:, split_index]

    data.num_classes = int(data.y.max().item()) + 1
    dataset.data = data
    return dataset


@torch.no_grad()
def prepare_propagation_graph(data: Data) -> Data:
    """Build ACSG random-walk propagation and the Table 6 base spectrum."""
    num_nodes = int(data.num_nodes)
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes, reduce="mean")
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)

    row = edge_index[0]
    node_degree = degree(row, num_nodes=num_nodes, dtype=torch.float64)
    if num_nodes < 2 or torch.any(node_degree <= 0):
        raise ValueError("The propagation graph must be connected and have at least two nodes.")

    edge_weight = node_degree[row].reciprocal().to(dtype=data.x.dtype)
    row = row.detach().cpu().numpy()
    col = edge_index[1].detach().cpu().numpy()
    adjacency = coo_matrix(
        (np.ones(row.shape[0], dtype=np.float64), (row, col)),
        shape=(num_nodes, num_nodes),
    ).tocsr()
    adjacency = 0.5 * (adjacency + adjacency.T)

    component_count, _ = connected_components(adjacency, directed=False)
    if component_count != 1:
        raise ValueError(f"Propagation graph has {component_count} connected components.")

    weighted_degree = adjacency.sum(axis=1).A1
    laplacian = diags(weighted_degree) - adjacency
    lambda2 = float(sorted(eigsh(laplacian, k=2, which="SM", return_eigenvectors=False))[1])
    lambda_max = float(eigsh(laplacian, k=1, which="LA", return_eigenvectors=False)[0])
    if lambda2 <= 1e-10 or lambda_max <= 0.0:
        raise ValueError("Invalid Laplacian spectrum for the propagation graph.")

    data.edge_index = edge_index
    data.edge_attr = edge_weight
    data.edge_weight = edge_weight
    data.degrees = node_degree.to(dtype=data.x.dtype)
    data.lambda2 = lambda2
    data.lambda_max = lambda_max
    return data
