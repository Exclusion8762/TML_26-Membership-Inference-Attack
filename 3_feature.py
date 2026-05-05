def get_three_feature_scores(dataset, model, device, batch_size=256):
    def collate_fn(batch):
        ids    = [b[0] for b in batch]
        imgs   = torch.stack([b[1] for b in batch])
        labels = torch.tensor([b[2] for b in batch])
        return ids, imgs, labels

    loader = DataLoader(dataset, batch_size=batch_size,
                       shuffle=False, collate_fn=collate_fn)

    all_ids = []
    all_conf, all_negloss, all_entropy = [], [], []

    with torch.no_grad():
        for ids, imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            probs  = F.softmax(logits, dim=1)

            # Feature 1 — Confidence on true label
            true_probs = probs[range(len(labels)), labels]
            all_conf.extend(true_probs.cpu().numpy())

            # Feature 2 — Negative cross-entropy loss
            loss = F.cross_entropy(logits, labels, reduction='none')
            all_negloss.extend((-loss).cpu().numpy())

            # Feature 3 — Negative entropy
            entropy = -(probs * torch.log(probs.clamp(1e-7))).sum(dim=1)
            all_entropy.extend((-entropy).cpu().numpy())

            all_ids.extend([str(i) for i in ids])

    # Normalize each to [0,1] and combine
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-12)

    conf     = normalize(np.array(all_conf))
    negloss  = normalize(np.array(all_negloss))
    entropy  = normalize(np.array(all_entropy))
    combined = conf + negloss + entropy

    return all_ids, combined


# Validate on pub.pt
pub_ids, pub_scores = get_three_feature_scores(pub_ds, model, device)
true_labels = [int(m) for m in pub_ds.membership]

fpr, tpr, _ = roc_curve(true_labels, pub_scores)
idx = np.searchsorted(fpr, 0.05)
print(f"Three-feature TPR@5%FPR = {tpr[idx]:.4f}")