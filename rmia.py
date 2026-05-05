def get_conf(dataset, model, batch_size=256):
    """Get confidence scores on true labels for all samples."""
    def collate_fn(batch):
        ids = [b[0] for b in batch]
        imgs = torch.stack([b[1] for b in batch])
        labels = torch.tensor([b[2] for b in batch])
        return ids, imgs, labels

    loader = DataLoader(dataset, batch_size=batch_size,
                       shuffle=False, collate_fn=collate_fn)
    all_ids, all_conf, all_labels = [], [], []

    with torch.no_grad():
        for ids, imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            true_probs = probs[range(len(labels)), labels]
            all_ids.extend([str(i) for i in ids])
            all_conf.extend(true_probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_ids, np.array(all_conf), np.array(all_labels)


def rmia_score(target_conf, target_labels, 
               pop_conf, pop_labels, gamma=1.0):
    """
    RMIA offline without reference models.
    For each target sample x, compute fraction of population
    samples z where conf(x)/conf(z) > gamma.
    Optionally restrict z to same class for better calibration.
    """
    scores = []
    for i, (cx, lx) in enumerate(zip(target_conf, target_labels)):
        # Use population samples of SAME class as z
        same_class_mask = (pop_labels == lx)
        z_confs = pop_conf[same_class_mask]
        
        if len(z_confs) == 0:
            scores.append(0.5)
            continue
        
        # Fraction of z where cx/cz > gamma  →  cx > gamma * cz
        dominated = np.sum(cx > gamma * z_confs)
        scores.append(dominated / len(z_confs))
    
    return np.array(scores)


# ── GET CONFIDENCE SCORES ────────────────────────────────────────────
print("Scoring pub.pt...")
pub_ids, pub_conf, pub_label_arr = get_conf(pub_ds, model)
true_labels = [int(m) for m in pub_ds.membership]

# ── RUN RMIA ON pub.pt ───────────────────────────────────────────────
print("Running RMIA...")

# Use pub.pt non-members as population Z (we know who they are)
nonmember_mask = np.array(true_labels) == 0
pop_conf   = pub_conf[nonmember_mask]
pop_labels = pub_label_arr[nonmember_mask]

rmia_scores = rmia_score(pub_conf, pub_label_arr,
                         pop_conf, pop_labels, gamma=1.0)

fpr, tpr, _ = roc_curve(true_labels, rmia_scores)
idx = np.searchsorted(fpr, 0.05)
print(f"RMIA TPR@5%FPR on pub.pt = {tpr[idx]:.4f}")

m_scores  = rmia_scores[np.array(true_labels) == 1]
nm_scores = rmia_scores[np.array(true_labels) == 0]
print(f"Members     → mean: {np.mean(m_scores):.4f}")
print(f"Non-members → mean: {np.mean(nm_scores):.4f}")
print(f"Gap         → {np.mean(m_scores) - np.mean(nm_scores):.4f}")



print("\nScoring priv.pt...")
priv_ids, priv_conf, priv_label_arr = get_conf(priv_ds, model)
# Use ALL pub.pt as population for priv.pt scoring
priv_rmia = rmia_score(priv_conf, priv_label_arr,
                       pub_conf, pub_label_arr, gamma=1.0)
priv_rmia = (priv_rmia - priv_rmia.min()) / \
            (priv_rmia.max() - priv_rmia.min())
df = pd.DataFrame({"id": priv_ids, "score": priv_rmia})
df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)