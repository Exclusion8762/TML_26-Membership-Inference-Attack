import argparse
import numpy as np
import torch
from pathlib import Path
import torchvision.transforms as transforms

from lira_utils import (build_model, get_logit_scores,
                         gaussian_logpdf, tpr_at_fpr, MEAN, STD)

from torch.utils.data import Dataset

class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


class ScoreOnlyDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        item = self.base_dataset[index]
        if len(item) == 4:
            id_, img, label, _ = item
        else:
            id_, img, label = item
        return id_, img, label
        
def compute_lira_scores_pub(target_logits, shadow_logits, in_masks):
    """
    Proper per-sample LiRA score computation.

    For each sample x:
      - confs_in  = shadow scores when x WAS in shadow training set
      - confs_out = shadow scores when x was NOT in shadow training set
      - Fit Gaussians → compute log-likelihood ratio

    target_logits:  [n_pub]            logit scores from target model
    shadow_logits:  [n_shadows, n_pub] logit scores from each shadow
    in_masks:       [n_shadows, n_pub] True if sample i was IN shadow s
    """
    n_shadows, n_pub = shadow_logits.shape
    lira_scores   = np.zeros(n_pub, dtype=np.float64)
    fallback_count = 0

    # Global fallback stats (used when a sample has too few IN or OUT scores)
    global_mu_in   = np.mean(shadow_logits[in_masks])
    global_sig_in  = np.std(shadow_logits[in_masks])  + 1e-6
    global_mu_out  = np.mean(shadow_logits[~in_masks])
    global_sig_out = np.std(shadow_logits[~in_masks]) + 1e-6

    for i in range(n_pub):
        obs       = float(target_logits[i])
        in_mask_i = in_masks[:, i]           # which shadows had x IN
        c_in      = shadow_logits[in_mask_i,  i]
        c_out     = shadow_logits[~in_mask_i, i]

        if len(c_in) < 10 or len(c_out) < 10:
            # Fallback to global stats
            mu_in, sig_in   = global_mu_in,  global_sig_in
            mu_out, sig_out = global_mu_out, global_sig_out
            fallback_count += 1
        else:
            mu_in,  sig_in  = np.mean(c_in),  np.std(c_in)  + 1e-6
            mu_out, sig_out = np.mean(c_out), np.std(c_out) + 1e-6

        log_p_in  = gaussian_logpdf(obs, mu_in,  sig_in)
        log_p_out = gaussian_logpdf(obs, mu_out, sig_out)
        lira_scores[i] = log_p_in - log_p_out

    print(f"  Per-sample LiRA: {n_pub - fallback_count}/{n_pub} used individual Gaussians")
    print(f"  Fallback to global stats: {fallback_count}/{n_pub}")
    return lira_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shadow-dir', type=str, default='./shadow_data')
    parser.add_argument('--base-dir',   type=str, default='.')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    shadow_dir = Path(args.shadow_dir)
    base       = Path(args.base_dir)
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load shadow data
    print("Loading shadow data...")
    shadow_logits  = np.load(shadow_dir / 'pub_logits.npy')    # [n_shadows, n_pub]
    in_masks       = np.load(shadow_dir / 'in_masks.npy')      # [n_shadows, n_pub]
    pub_membership = np.load(shadow_dir / 'pub_membership.npy')
    print(f"  Shadows: {shadow_logits.shape[0]}, Pub samples: {shadow_logits.shape[1]}")
    print(f"  Avg IN fraction per shadow: {in_masks.mean():.3f}")

    # Get target model logit scores on pub.pt
    print("\nLoading target model...")
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    pub_ds = torch.load(base / 'pub.pt', weights_only=False)
    pub_ds.transform = transform

    target_model = build_model()
    target_model.load_state_dict(torch.load(base / 'model.pt', map_location='cpu'))
    target_model = target_model.to(device)

    print("Scoring pub.pt with target model...")
    _, target_logits, _ = get_logit_scores(
        target_model, pub_ds, device, batch_size=args.batch_size
    )

    # Compute proper per-sample LiRA scores
    print("\nComputing per-sample LiRA scores...")
    lira_scores = compute_lira_scores_pub(target_logits, shadow_logits, in_masks)

    # Evaluate
    tpr = tpr_at_fpr(lira_scores, pub_membership, fpr_target=0.05)
    print(f"\n{'='*40}")
    print(f"LiRA TPR@5%FPR on pub.pt = {tpr:.4f}")
    print(f"{'='*40}")

    # Distribution stats
    m_scores  = lira_scores[pub_membership == 1]
    nm_scores = lira_scores[pub_membership == 0]
    print(f"Members     → mean: {np.mean(m_scores):.4f}, std: {np.std(m_scores):.4f}")
    print(f"Non-members → mean: {np.mean(nm_scores):.4f}, std: {np.std(nm_scores):.4f}")
    print(f"Gap         → {np.mean(m_scores) - np.mean(nm_scores):.4f}")

    # Save for reference
    np.save(shadow_dir / 'pub_lira_scores.npy', lira_scores)
    print(f"\nSaved scores → {shadow_dir / 'pub_lira_scores.npy'}")
    print("\nIf TPR@5%FPR looks good, proceed to Job 3.")


if __name__ == '__main__':
    main()