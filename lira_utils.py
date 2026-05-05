import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18

MEAN = [0.7406, 0.5331, 0.7059]
STD  = [0.1491, 0.1864, 0.1301]
NUM_CLASSES = 9


def build_model():
    m = resnet18(weights=None)
    m.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = torch.nn.Identity()
    m.fc = torch.nn.Linear(512, NUM_CLASSES)
    return m


def get_logit_scores(model, dataset, device, batch_size=256):
    """
    Returns logit-scaled confidence scores φ(p) = log(p / (1 - p))
    on the true label for every sample.
    """
    def collate_fn(batch):
        ids = [b[0] for b in batch]
        imgs = torch.stack([b[1] for b in batch])
        labels = torch.tensor([b[2] for b in batch])
        return ids, imgs, labels

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    all_ids, all_logits, all_labels = [], [], []

    model.eval()
    with torch.no_grad():
        for ids, imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            probs = F.softmax(model(imgs), dim=1)
            p = probs[range(len(labels)), labels].clamp(1e-7, 1 - 1e-7)
            phi = torch.log(p / (1 - p)).cpu().numpy()

            all_ids.extend([int(i) for i in ids])
            all_logits.extend(phi)
            all_labels.extend(labels.cpu().numpy())

    return (
        np.array(all_ids),
        np.array(all_logits, dtype=np.float32),
        np.array(all_labels, dtype=np.int64),
    )


def gaussian_logpdf(x, mu, sigma):
    sigma = max(float(sigma), 1e-6)
    z = (x - mu) / sigma
    return -0.5 * z * z - np.log(sigma) - 0.5 * np.log(2 * np.pi)


def roc_curve_numpy(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("y_true and y_score must be 1D arrays.")
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")

    pos = y_true == 1
    neg = y_true == 0
    P = int(np.sum(pos))
    N = int(np.sum(neg))

    if P == 0 or N == 0:
        raise ValueError("Both positive and negative samples are required.")

    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    distinct = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct, len(y_score_sorted) - 1]

    tps = np.cumsum(y_true_sorted == 1)[threshold_idxs]
    fps = np.cumsum(y_true_sorted == 0)[threshold_idxs]

    tpr = tps / P
    fpr = fps / N

    thresholds = y_score_sorted[threshold_idxs]

    # Match sklearn's ROC shape a bit more closely.
    tpr = np.r_[0.0, tpr, 1.0]
    fpr = np.r_[0.0, fpr, 1.0]
    thresholds = np.r_[np.inf, thresholds, -np.inf]

    return fpr, tpr, thresholds


def tpr_at_fpr(scores, memberships, fpr_target=0.05):
    fpr, tpr, thresholds = roc_curve_numpy(memberships, scores)
    idx = np.searchsorted(fpr, fpr_target, side="right") - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx])
