"""Dataset loaders needed by the main heterophilic-graph experiment."""

import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import remove_self_loops, to_undirected
from torch_sparse import SparseTensor, coalesce


class Actor(InMemoryDataset):
    url = "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            "out1_node_feature_label.txt",
            "out1_graph_edges.txt",
            *[f"film_split_0.6_0.2_{index}.npz" for index in range(10)],
        ]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        for filename in self.raw_file_names[:2]:
            download_url(f"{self.url}/new_data/film/{filename}", self.raw_dir)
        for filename in self.raw_file_names[2:]:
            download_url(f"{self.url}/splits/{filename}", self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], "r") as stream:
            rows = [line.split("\t") for line in stream.read().split("\n")[1:-1]]

        feature_rows, feature_columns = [], []
        labels = torch.empty(len(rows), dtype=torch.long)
        for node_id, columns, label in rows:
            columns = [int(column) for column in columns.split(",")]
            feature_rows.extend([int(node_id)] * len(columns))
            feature_columns.extend(columns)
            labels[int(node_id)] = int(label)
        features = SparseTensor(
            row=torch.tensor(feature_rows),
            col=torch.tensor(feature_columns),
        ).to_dense()

        with open(self.raw_paths[1], "r") as stream:
            edges = stream.read().split("\n")[1:-1]
        edge_index = torch.tensor(
            [[int(value) for value in edge.split("\t")] for edge in edges],
            dtype=torch.long,
        ).t().contiguous()
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index)
        edge_index, _ = coalesce(
            edge_index,
            None,
            features.size(0),
            features.size(0),
        )

        train_masks, validation_masks, test_masks = [], [], []
        for filename in self.raw_paths[2:]:
            split = np.load(filename)
            train_masks.append(torch.from_numpy(split["train_mask"]).bool())
            validation_masks.append(torch.from_numpy(split["val_mask"]).bool())
            test_masks.append(torch.from_numpy(split["test_mask"]).bool())

        data = Data(
            x=features,
            edge_index=edge_index,
            y=labels,
            train_mask=torch.stack(train_masks, dim=1),
            val_mask=torch.stack(validation_masks, dim=1),
            test_mask=torch.stack(test_masks, dim=1),
        )
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])


class WebKB(InMemoryDataset):
    url = "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/new_data"

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        if self.name not in {"cornell", "texas", "wisconsin"}:
            raise ValueError(f"Unsupported WebKB dataset: {name}")
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        return ["out1_node_feature_label.txt", "out1_graph_edges.txt"]

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        for filename in self.raw_file_names:
            download_url(f"{self.url}/{self.name}/{filename}", self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], "r") as stream:
            rows = stream.read().split("\n")[1:-1]
        features = torch.tensor(
            [[float(value) for value in row.split("\t")[1].split(",")] for row in rows],
            dtype=torch.float32,
        )
        labels = torch.tensor(
            [int(row.split("\t")[2]) for row in rows],
            dtype=torch.long,
        )

        with open(self.raw_paths[1], "r") as stream:
            edges = stream.read().split("\n")[1:-1]
        edge_index = torch.tensor(
            [[int(value) for value in edge.split("\t")] for edge in edges],
            dtype=torch.long,
        ).t().contiguous()
        edge_index = to_undirected(edge_index)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = coalesce(
            edge_index,
            None,
            features.size(0),
            features.size(0),
        )

        data = Data(x=features, edge_index=edge_index, y=labels)
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f"{self.name}()"
