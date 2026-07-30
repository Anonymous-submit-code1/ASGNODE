import torch


@torch.no_grad()
def evaluate_logits(logits, data, metric="accuracy"):
    if metric != "accuracy":
        raise ValueError("The main experiment reports node-classification accuracy only.")

    predictions = logits.argmax(dim=-1)
    labels = data.y.view(-1)
    return [
        float((predictions[mask] == labels[mask]).float().mean().item())
        for mask in (data.train_mask, data.val_mask, data.test_mask)
    ]
