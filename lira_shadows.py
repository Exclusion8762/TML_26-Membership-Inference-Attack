import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from lira_utils import build_model, get_logit_scores, MEAN, STD


def train_shadow(train_dataset, device, epochs, batch_size, lr):
    """
    Train one shadow model.
    Key fix from original: SGD + cosine LR instead of Adam.
    SGD with momentum causes proper memorization needed for LiRA.
    """
    def collate_fn(batch):
        ids    = [b[0] for b in batch]
        imgs   = torch.stack([b[1] for b in batch])
        labels = torch.tensor([b[2] for b in batch])
        return ids, imgs, labels

    model     = build_model().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr,
        momentum=0.9, weight_decay=5e-4, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    criterion = nn.CrossEntropyLoss()
    loader    = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, collate_fn=collate_fn,
        num_workers=4, pin_memory=True
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for ids, imgs, labels in loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            avg = total_loss / len(loader)
            lr_now = scheduler.get_last_lr()[0]
            print(f"    epoch {epoch+1}/{epochs}  "
                  f"loss={avg:.4f}  lr={lr_now:.6f}")

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-shadows',  type=int,   default=64)
    parser.add_argument('--epochs',       type=int,   default=100)
    parser.add_argument('--batch-size',   type=int,   default=256)
    parser.add_argument('--lr',           type=float, default=0.1)
    parser.add_argument('--base-dir',     type=str,   default='.')
    parser.add_argument('--output-dir',   type=str,   default='./shadow_data')
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--score-priv',   action='store_true',
                        help='Also score priv.pt during shadow training')
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base       = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device:       {device}")
    print(f"Num shadows:  {args.num_shadows}")
    print(f"Epochs/shadow:{args.epochs}")
    print(f"Output dir:   {output_dir}")

    # Load datasets
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    print("\nLoading pub.pt...")

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

    pub_ds = torch.load(base / 'pub.pt', weights_only=False)
    pub_ds.transform = transform
    n_pub = len(pub_ds)

    priv_ds = None
    n_priv  = 0
    if args.score_priv:
        print("Loading priv.pt...")
        priv_ds = torch.load(base / 'priv.pt', weights_only=False)
        priv_ds.transform = transform
        n_priv = len(priv_ds)

    # Pre-allocate score arrays
    all_pub_logits = np.zeros((args.num_shadows, n_pub),   dtype=np.float32)
    all_in_masks   = np.zeros((args.num_shadows, n_pub),   dtype=bool)
    if args.score_priv:
        all_priv_logits = np.zeros((args.num_shadows, n_priv), dtype=np.float32)

    rng = np.random.default_rng(args.seed)

    # ── Main shadow training loop ────────────────────────────────────
    for s in range(args.num_shadows):
        print(f"\n{'='*55}")
        print(f"Shadow {s+1}/{args.num_shadows}")

        # Random 50% split — proper LiRA requirement
        perm     = rng.permutation(n_pub)
        n_in     = n_pub // 2
        in_idx   = perm[:n_in]
        in_mask  = np.zeros(n_pub, dtype=bool)
        in_mask[in_idx] = True
        all_in_masks[s] = in_mask

        # Train on IN subset only
        train_ds     = Subset(pub_ds, in_idx.tolist())
        shadow_model = train_shadow(
            train_ds, device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )

        # Score ALL pub.pt samples (both IN and OUT)
        print("  Collecting pub.pt logit scores...")
        _, pub_logits, _ = get_logit_scores(
            shadow_model, pub_ds, device, batch_size=args.batch_size
        )
        all_pub_logits[s] = pub_logits

        # Score priv.pt (all OUT by definition)
        if args.score_priv:
            print("  Collecting priv.pt logit scores...")
            _, priv_logits, _ = get_logit_scores(
                shadow_model, priv_ds, device, batch_size=args.batch_size
            )
            all_priv_logits[s] = priv_logits

        # Cleanup GPU
        del shadow_model
        torch.cuda.empty_cache()

        # ── Incremental save after every shadow ───────────────────────
        # If cluster job crashes, you don't lose all progress
        np.save(output_dir / 'pub_logits.npy',  all_pub_logits[:s+1])
        np.save(output_dir / 'in_masks.npy',    all_in_masks[:s+1])
        if args.score_priv:
            np.save(output_dir / 'priv_logits.npy', all_priv_logits[:s+1])
        print(f"  Saved checkpoint after shadow {s+1}")

    # ── Save pub.pt metadata ─────────────────────────────────────────
    print("\nSaving pub.pt metadata...")
    pub_labels     = np.array([int(pub_ds.labels[i])     for i in range(n_pub)])
    pub_ids        = np.array([int(pub_ds.ids[i])        for i in range(n_pub)])
    pub_membership = np.array([int(pub_ds.membership[i]) for i in range(n_pub)])
    np.save(output_dir / 'pub_labels.npy',     pub_labels)
    np.save(output_dir / 'pub_ids.npy',        pub_ids)
    np.save(output_dir / 'pub_membership.npy', pub_membership)

    if args.score_priv:
        print("Saving priv.pt metadata...")
        priv_labels = np.array([int(priv_ds.labels[i]) for i in range(n_priv)])
        priv_ids    = np.array([int(priv_ds.ids[i])    for i in range(n_priv)])
        np.save(output_dir / 'priv_labels.npy', priv_labels)
        np.save(output_dir / 'priv_ids.npy',    priv_ids)

    print(f"\nJob 1 complete. All data saved to: {output_dir}")


if __name__ == '__main__':
    main()