# Membership Inference Attack — TML 2026

Repository for the Membership Inference Attack assignment from the 
Trustworthy Machine Learning course (2026).

## Best Result

**TPR@5%FPR: 0.05232** — achieved using LiRA (Carlini et al., 2022)

## Repository Structure

| File | Description |
|---|---|
| `conf.py` | Confidence-based attack |
| `3_feature.py` | Three-feature attack (confidence + neg-loss + entropy) |
| `rmia.py` | RMIA attack (Zarifzadeh et al., 2023) |
| `lira_utils.py` | Shared utilities (model, logit scoring, Gaussian functions) |
| `lira_shadows.py` | Train shadow models + collect scores |
| `lira_pub_eval.py` | Evaluate LiRA on pub.pt (local validation) |
| `lira_sub.py` | Score priv.pt + generate submission.csv |
| `README.md` | This file |

## Reproducing Best Result (LiRA)

### Requirements

```bash
pip install torch torchvision scikit-learn pandas numpy
```

### Step 1 — Train Shadow Models

```bash
python lira_shadows.py \
    --num-shadows 64 \
    --epochs 100 \
    --lr 0.1 \
    --base-dir /path/to/data \
    --output-dir ./shadow_data \
    --score-priv
```

### Step 2 — Validate on pub.pt

```bash
python lira_pub_eval.py \
    --shadow-dir ./shadow_data \
    --base-dir /path/to/data
```

### Step 3 — Generate Submission

```bash
python lira_sub.py \
    --shadow-dir ./shadow_data \
    --base-dir /path/to/data \
    --output submission.csv
```

## Other Attacks

Each attack file (`conf.py`, `3_feature.py`, `rmia.py`) contains 
the attack logic that must be added to `task_template.py` to access 
the pre-trained model and datasets before running.

## References

- Carlini et al., *Membership Inference Attacks From First Principles*, IEEE S&P 2022
- Zarifzadeh et al., *Low-Cost High-Power Membership Inference Attacks*, ICML 2024
- Shokri et al., *Membership Inference Attacks Against Machine Learning Models*, IEEE S&P 2017
