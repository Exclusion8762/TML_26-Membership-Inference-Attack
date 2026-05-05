import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from lira_utils import build_model, get_logit_scores, gaussian_logpdf, MEAN, STD

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


def estimate_in_out_delta(shadow_logits, in_masks):
    """
    Estimate the average gap between IN and OUT logit scores
    from pub.pt shadow data. Used to approximate IN distribution
    for priv.pt samples (which are always OUT).
    """
    deltas = []
    for s in range(shadow_logits.shape[0]):
        in_scores  = shadow_logits[s,  in_masks[s]]
        out_scores = shadow_logits[s, ~in_masks[s]]
        if len(in_scores) > 0 and len(out_scores) > 0:
            deltas.append(np.mean(in_scores) - np.mean(out_scores))
    delta = float(np.mean(deltas)) if deltas else 0.5
    print(f"  Estimated IN/OUT delta from pub.pt: {delta:.4f}")
    return delta


def compute_lira_scores_priv(target_logits, priv_shadow_logits,
                              pub_shadow_logits, in_masks):
    """
    Offline LiRA for priv.pt.

    For each priv.pt sample:
      - All shadow scores are OUT (priv.pt was never in shadow training)
      - μ_out, σ_out  → fit from shadow OUT scores on priv.pt
      - μ_in          → approximate as μ_out + delta
      - σ_in          → same as σ_out (conservative)
      - Score         → log p(obs | IN) - log p(obs | OUT)
    """
    n_priv = len(target_logits)
    delta  = estimate_in_out_delta(pub_shadow_logits, in_masks)

    lira_scores = np.zeros(n_priv, dtype=np.float64)

    for i in range(n_priv):
        obs       = float(target_logits[i])
        out_scores = priv_shadow_logits[:, i]  # all OUT

        if len(out_scores) < 2:
            lira_scores[i] = obs
            continue

        mu_out  = np.mean(out_scores)
        sig_out = np.std(out_scores) + 1e-6

        # Offline IN approximation
        mu_in  = mu_out + delta
        sig_in = sig_out  # conservative: same spread

        log_p_in  = gaussian_logpdf(obs, mu_in,  sig_in)
        log_p_out = gaussian_logpdf(obs, mu_out, sig_out)
        lira_scores[i] = log_p_in - log_p_out

    return lira_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shadow-dir', type=str, default='./shadow_data')
    parser.add_argument('--base-dir',   type=str, default='.')
    parser.add_argument('--output',     type=str, default='submission.csv')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    shadow_dir = Path(args.shadow_dir)
    base       = Path(args.base_dir)
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Verify required files exist
    for fname in ['pub_logits.npy', 'in_masks.npy',
                  'priv_logits.npy', 'priv_ids.npy']:
        if not (shadow_dir / fname).exists():
            raise FileNotFoundError(
                f"Missing {fname}. Run job1 with --score-priv first!"
            )

    # Load shadow data
    print("Loading shadow data...")
    pub_shadow_logits  = np.load(shadow_dir / 'pub_logits.npy')   # [n_shadows, n_pub]
    in_masks           = np.load(shadow_dir / 'in_masks.npy')     # [n_shadows, n_pub]
    priv_shadow_logits = np.load(shadow_dir / 'priv_logits.npy')  # [n_shadows, n_priv]
    priv_ids           = np.load(shadow_dir / 'priv_ids.npy')     # [n_priv]
    print(f"  Shadows: {pub_shadow_logits.shape[0]}")
    print(f"  Priv samples: {priv_shadow_logits.shape[1]}")

    # Score priv.pt with target model
    print("\nLoading target model...")
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    priv_ds = torch.load(base / 'priv.pt', weights_only=False)
    priv_ds.transform = transform

    target_model = build_model()
    target_model.load_state_dict(
        torch.load(base / 'model.pt', map_location='cpu')
    )
    target_model = target_model.to(device)

    print("Scoring priv.pt with target model...")
    _, target_logits, _ = get_logit_scores(
        target_model, priv_ds, device, batch_size=args.batch_size
    )

    # Compute offline LiRA scores
    print("\nComputing offline LiRA scores for priv.pt...")
    lira_scores = compute_lira_scores_priv(
        target_logits, priv_shadow_logits,
        pub_shadow_logits, in_masks
    )

    # Normalize to [0, 1]
    lo, hi = lira_scores.min(), lira_scores.max()
    normalized = (lira_scores - lo) / (hi - lo + 1e-12)

    # Build and validate submission
    df = pd.DataFrame({
        'id':    [str(int(i)) for i in priv_ids],
        'score': normalized.astype(float)
    })
    assert df['id'].nunique()        == len(df),  "Duplicate IDs!"
    assert df['score'].between(0,1).all(),         "Scores out of range!"
    assert not df.isnull().any().any(),            "Missing values!"

    df.to_csv(args.output, index=False)
    print(f"\nSubmission saved → {args.output}")
    print(f"Score stats: min={normalized.min():.4f}  "
          f"max={normalized.max():.4f}  mean={normalized.mean():.4f}")


if __name__ == '__main__':
    main()