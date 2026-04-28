from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from models.model_clam import CLAM_MB


def load_clam_mb(ckpt_path: Path, device: str, embed_dim: int, n_classes: int = 2):
    from topk.svm import SmoothTop1SVM

    instance_loss_fn = SmoothTop1SVM(n_classes=2)
    model = CLAM_MB(
        gate=True,
        size_arg="small",
        dropout=0.0,
        k_sample=8,
        n_classes=n_classes,
        embed_dim=embed_dim,
        instance_loss_fn=instance_loss_fn,
    )
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("CLAM load missing:", missing)
    print("CLAM load unexpected:", unexpected)
    model.eval().to(device)
    return model


@torch.no_grad()
def clam_predict_bag(model, bag_feats_np: np.ndarray, device: str):
    h = torch.from_numpy(bag_feats_np).float().to(device)
    logits, prob, yhat, araw, _ = model(h, return_features=False)
    att = araw.detach().float().cpu().numpy()
    att = att.reshape(-1)
    return logits.detach().cpu().numpy(), prob.detach().cpu().numpy(), int(yhat.item()), att
