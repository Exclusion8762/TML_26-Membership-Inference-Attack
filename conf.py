import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve

def get_scores(dataset, model, device, batch_size=256):
    def collate_fn(batch):
        ids    = [b[0] for b in batch]
        imgs   = torch.stack([b[1] for b in batch])
        labels = torch.tensor([b[2] for b in batch])
        return ids, imgs, labels

    loader = DataLoader(dataset, batch_size=batch_size,
                       shuffle=False, collate_fn=collate_fn)
    all_ids, all_scores = [], []

    with torch.no_grad():
        for ids, imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            probs  = F.softmax(model(imgs), dim=1)

            # True label confidence
            true_probs = probs[range(len(labels)), labels]

            all_ids.extend([str(i) for i in ids])
            all_scores.extend(logit_scores.cpu().numpy())

    return all_ids, np.array(all_scores)


# Validate on pub.pt
pub_ids, pub_scores = get_scores(pub_ds, model, device)
true_labels = [int(m) for m in pub_ds.membership]

fpr, tpr, _ = roc_curve(true_labels, pub_scores)
idx = np.searchsorted(fpr, 0.05)
print(f"Confidence TPR@5%FPR = {tpr[idx]:.4f}")