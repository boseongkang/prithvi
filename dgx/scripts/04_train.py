"""
04_train.py

SegFormer-B2 fault detection training.

Input:  data/patches/{train,val,test}/{images,masks}/patch_XXXXX.npy
Output: runs/<run_id>/best_model.pt
       runs/<run_id>/metrics.json
       runs/<run_id>/train_log.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation


# ── Dataset ──────────────────────────────────────────────────

class FaultDataset(Dataset):
    def __init__(self, split_dir, mean, std, augment=False):
        self.image_dir = Path(split_dir) / "images"
        self.mask_dir = Path(split_dir) / "masks"
        self.files = sorted(self.image_dir.glob("*.npy"))
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx].name

        img = np.load(self.image_dir / fname).astype(np.float32)
        mask = np.load(self.mask_dir / fname).astype(np.int64)

        if img.ndim == 2:
            img = img[np.newaxis]
        elif img.ndim == 3 and img.shape[-1] in (1, 2, 3):
            img = img.transpose(2, 0, 1)

        for c in range(img.shape[0]):
            img[c] = (img[c] - self.mean[c]) / (self.std[c] + 1e-8)

        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = np.clip(mask, 0, 1)

        if self.augment:
            # Random horizontal flip (W axis), vertical flip (H axis), and 0/90/180/270° rotation.
            # Applied identically to image (C,H,W) and mask (H,W) so labels stay aligned.
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(img[:, :, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1, :])
                mask = np.ascontiguousarray(mask[::-1, :])
            k = int(np.random.randint(0, 4))
            if k:
                img = np.ascontiguousarray(np.rot90(img, k=k, axes=(1, 2)))
                mask = np.ascontiguousarray(np.rot90(mask, k=k, axes=(0, 1)))

        return torch.from_numpy(img), torch.from_numpy(mask)


# ── Loss ─────────────────────────────────────────────────────

class WeightedCEDiceLoss(nn.Module):
    def __init__(self, fault_weight=5.0, dice_weight=1.0, ce_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, fault_weight]))
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)

        probs = F.softmax(logits, dim=1)[:, 1]
        targets_f = targets.float()
        intersection = (probs * targets_f).sum()
        dice = 1 - (2 * intersection + 1) / (probs.sum() + targets_f.sum() + 1)

        return self.ce_weight * ce_loss + self.dice_weight * dice


# ── Metrics ──────────────────────────────────────────────────

# Tolerance IoU: dilate pred and gt by this many pixels before scoring.
# Justified for 1m buffer (2px label): a 1-px lateral error halves pixel-IoU,
# and there's a structural ~3–10px offset between QFault trace (toe) and
# scarp ridge (the visible signal). 3px ≈ 3m absorbs both without blurring
# distinct traces (smallest separation in this region ≫ 6m).
TOLERANCE_PX = 3


def dilate_binary(mask, radius=TOLERANCE_PX):
    """Morphological dilation via max-pool. Accepts float tensor (B,H,W) or
    (B,1,H,W) with 0/1 values; returns same shape, still 0/1."""
    squeeze = (mask.dim() == 3)
    if squeeze:
        mask = mask.unsqueeze(1)
    k = 2 * radius + 1
    out = F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius)
    return out.squeeze(1) if squeeze else out


def compute_metrics(preds, targets):
    """preds and targets are 1D boolean/int arrays."""
    tp = int(((preds == 1) & (targets == 1)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())

    iou_fault = tp / (tp + fp + fn + 1e-8)
    iou_bg = tn / (tn + fp + fn + 1e-8)
    miou = (iou_fault + iou_bg) / 2
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "iou_fault": round(iou_fault, 4),
        "iou_bg": round(iou_bg, 4),
        "miou": round(miou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ── Model ────────────────────────────────────────────────────

def _find_first_patch_proj(model):
    """Find the first patch embedding Conv2d regardless of transformers version."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and "patch" in name and name.endswith(".proj"):
            return name, module
    raise RuntimeError("Cannot find patch embedding proj layer in model")


def _set_nested_attr(model, dotted_name, new_module):
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def build_model(num_channels=2, num_labels=2, dropout=0.1):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2",
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        classifier_dropout_prob=dropout,
    )

    if num_channels != 3:
        proj_name, old_proj = _find_first_patch_proj(model)
        new_proj = nn.Conv2d(
            num_channels, old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
        )
        with torch.no_grad():
            new_proj.weight[:, :min(num_channels, 3)] = old_proj.weight[:, :min(num_channels, 3)]
            new_proj.bias.copy_(old_proj.bias)
        _set_nested_attr(model, proj_name, new_proj)

    return model


# ── Optimizer + Scheduler ────────────────────────────────────

def build_optimizer(model, encoder_lr=1e-5, decoder_lr=1e-4, weight_decay=0.01):
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("decode_head."):
            head_params.append(param)
        else:
            backbone_params.append(param)

    return torch.optim.AdamW([
        {"params": backbone_params, "lr": encoder_lr},
        {"params": head_params, "lr": decoder_lr},
    ], weight_decay=weight_decay)


class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_steps, power=0.9, last_epoch=-1):
        self.total_steps = total_steps
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = min(self.last_epoch, self.total_steps)
        factor = (1 - step / self.total_steps) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]


# ── Data Stats ───────────────────────────────────────────────

def compute_channel_stats(split_dir, max_files=500):
    """Compute per-channel mean and std from a sample of patches (NaN-safe)."""
    image_dir = Path(split_dir) / "images"
    files = sorted(image_dir.glob("*.npy"))[:max_files]

    sums = None
    sq_sums = None
    counts = None

    for f in files:
        img = np.nan_to_num(np.load(f).astype(np.float64), nan=0.0)
        if img.ndim == 2:
            img = img[np.newaxis]
        elif img.ndim == 3 and img.shape[-1] in (1, 2, 3):
            img = img.transpose(2, 0, 1)

        c = img.shape[0]
        if sums is None:
            sums = np.zeros(c)
            sq_sums = np.zeros(c)
            counts = np.zeros(c)

        for ch in range(c):
            valid = img[ch][img[ch] != 0.0]
            sums[ch] += valid.sum()
            sq_sums[ch] += (valid ** 2).sum()
            counts[ch] += valid.size

    mean = sums / np.maximum(counts, 1)
    std = np.sqrt(sq_sums / np.maximum(counts, 1) - mean ** 2)
    return mean.tolist(), std.tolist()


# ── Threshold Sweep ──────────────────────────────────────────

def threshold_sweep(model, loader, device, thresholds=None):
    """Sweep thresholds in one inference pass. For each threshold we accumulate
    both pixel-IoU counts (tp/fp/fn/tn) and tolerance-IoU counts (intersection
    and union of the dilated masks). Returns (best_pixel_thresh,
    best_tol_thresh, per-threshold results dict)."""
    if thresholds is None:
        thresholds = [i * 0.05 for i in range(1, 11)]

    model.eval()
    pix = {t: [0, 0, 0, 0] for t in thresholds}   # tp, fp, fn, tn
    tol = {t: [0, 0] for t in thresholds}         # inter, union

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(pixel_values=images).logits
            logits_up = F.interpolate(logits, size=masks.shape[1:],
                                      mode="bilinear", align_corners=False)
            probs = F.softmax(logits_up, dim=1)[:, 1]
            gt = (masks == 1)
            gt_d = dilate_binary(gt.float()) > 0  # gt dilation reused across thresholds

            for t in thresholds:
                preds = (probs >= t)
                pix[t][0] += (preds & gt).sum().item()
                pix[t][1] += (preds & ~gt).sum().item()
                pix[t][2] += (~preds & gt).sum().item()
                pix[t][3] += (~preds & ~gt).sum().item()
                preds_d = dilate_binary(preds.float()) > 0
                tol[t][0] += (preds_d & gt_d).sum().item()
                tol[t][1] += (preds_d | gt_d).sum().item()

    results = {}
    best_pixel_thresh = thresholds[0]; best_pixel_iou = -1.0
    best_tol_thresh = thresholds[0]; best_tol_iou = -1.0

    for t in thresholds:
        tp, fp, fn, tn = pix[t]
        inter, union = tol[t]
        iou_fault = tp / (tp + fp + fn + 1e-8)
        iou_bg = tn / (tn + fp + fn + 1e-8)
        miou = (iou_fault + iou_bg) / 2
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        tol_iou = inter / (union + 1e-8) if union else 0.0

        results[f"{t:.2f}"] = {
            "iou_fault": round(iou_fault, 4),
            "tol_iou": round(tol_iou, 4),
            "iou_bg": round(iou_bg, 4),
            "miou": round(miou, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
        if iou_fault > best_pixel_iou:
            best_pixel_iou = iou_fault; best_pixel_thresh = t
        if tol_iou > best_tol_iou:
            best_tol_iou = tol_iou; best_tol_thresh = t

    return best_pixel_thresh, best_tol_thresh, results


# ── Train / Eval ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    n_batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(pixel_values=images).logits
        logits_up = F.interpolate(logits, size=masks.shape[1:],
                                  mode="bilinear", align_corners=False)

        loss = criterion(logits_up, masks)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    pix_tp = pix_fp = pix_fn = pix_tn = 0
    tol_inter = tol_union = 0  # IoU of (dilated pred) vs (dilated gt)

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(pixel_values=images).logits
        logits_up = F.interpolate(logits, size=masks.shape[1:],
                                  mode="bilinear", align_corners=False)

        loss = criterion(logits_up, masks)
        total_loss += loss.item()
        n_batches += 1

        probs = F.softmax(logits_up, dim=1)[:, 1]
        preds = (probs >= threshold)
        gt = (masks == 1)

        pix_tp += (preds & gt).sum().item()
        pix_fp += (preds & ~gt).sum().item()
        pix_fn += (~preds & gt).sum().item()
        pix_tn += (~preds & ~gt).sum().item()

        preds_d = dilate_binary(preds.float()) > 0
        gt_d = dilate_binary(gt.float()) > 0
        tol_inter += (preds_d & gt_d).sum().item()
        tol_union += (preds_d | gt_d).sum().item()

    iou_fault = pix_tp / (pix_tp + pix_fp + pix_fn + 1e-8)
    iou_bg = pix_tn / (pix_tn + pix_fp + pix_fn + 1e-8)
    miou = (iou_fault + iou_bg) / 2
    precision = pix_tp / (pix_tp + pix_fp + 1e-8)
    recall = pix_tp / (pix_tp + pix_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    tol_iou = tol_inter / (tol_union + 1e-8) if tol_union else 0.0

    return {
        "iou_fault": round(iou_fault, 4),
        "iou_bg": round(iou_bg, 4),
        "miou": round(miou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tol_iou": round(tol_iou, 4),
        "tp": pix_tp, "fp": pix_fp, "fn": pix_fn, "tn": pix_tn,
        "loss": round(total_loss / max(n_batches, 1), 6),
    }


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/patches")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--decoder-lr", type=float, default=1e-4)
    # fault_weight: positive-class weight in the CE term. With the 20m buffer
    # (02_process_faults.py: ~40px-wide label band at 1m DEM) the positive
    # ratio is ~1.27%, which sits in the same regime as the successful Owens
    # run (0.46-1.19% fault, weight 5-10). So default 5.0 and, if needed,
    # sweep 5 -> 10. The earlier 1m-buffer ablation paired ~0.09% sparsity with
    # weight 150 — that was the failure mode: such a high weight on so few
    # positives makes the model fire fault everywhere with no confidence
    # (precision ~0.02, predictions a huge blob or empty). The 20m buffer fixes
    # the sparsity at the source, so the weight no longer needs to be extreme.
    parser.add_argument("--fault-weight", type=float, default=5.0)
    # patience: val loss rose monotonically (1.31 -> 2.30 over 50 epochs) with
    # the best val IoU very early, so 15 just burned epochs past the optimum.
    # 8 stops much closer to the best checkpoint.
    parser.add_argument("--patience", type=int, default=8)
    # Regularization knobs to fight the observed overfitting. weight_decay was
    # already 0.01 inside the optimizer; expose it and bump the default. dropout
    # is the SegFormer classification-head dropout (classifier_dropout_prob).
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--channels", type=int, default=1)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data stats ──
    print("\nComputing channel statistics from training set...")
    mean, std = compute_channel_stats(data_dir / "train")
    print(f"  Mean: {mean}")
    print(f"  Std:  {std}")

    # ── Datasets ──
    # Augmentation on train only: random h/v flip + 90° rotation. The fault
    # geometry is rotation/reflection-invariant in plan view, and the prior run
    # showed sharp overfitting (best val IoU at epoch 1, train loss falling
    # while val loss rose), so these symmetry-preserving transforms expand the
    # effective training set without distorting the labels.
    train_ds = FaultDataset(data_dir / "train", mean, std, augment=True)
    val_ds = FaultDataset(data_dir / "val", mean, std, augment=False)
    test_ds = FaultDataset(data_dir / "test", mean, std, augment=False)

    print(f"\nDataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # ── Model ──
    print(f"\nBuilding SegFormer-B2 ({args.channels}ch input, 2 classes)...")
    model = build_model(num_channels=args.channels, num_labels=2, dropout=args.dropout).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Parameters: {n_params:.1f}M total, {n_train:.1f}M trainable")

    # ── Optimizer ──
    criterion = WeightedCEDiceLoss(fault_weight=args.fault_weight).to(device)
    optimizer = build_optimizer(model, encoder_lr=args.encoder_lr,
                                decoder_lr=args.decoder_lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = PolynomialLR(optimizer, total_steps=total_steps, power=0.9)

    # ── Config dump ──
    config = {
        "model": "nvidia/mit-b2",
        "channels": args.channels,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "encoder_lr": args.encoder_lr,
        "decoder_lr": args.decoder_lr,
        "fault_weight": args.fault_weight,
        "patience": args.patience,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "optimizer": "AdamW",
        "scheduler": "PolynomialLR(power=0.9)",
        "loss": "WeightedCE + Dice",
        "augmentation": "train: random h/v flip + 90° rotation",
        "mean": mean,
        "std": std,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "device": str(device),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Training loop ──
    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...")
    print(f"{'Epoch':>5s}  {'Train Loss':>10s}  {'Val Loss':>10s}  "
          f"{'Val IoU_f':>9s}  {'Val tIoU':>9s}  {'Val F1':>7s}  {'LR_enc':>10s}")
    print("-" * 80)

    train_log = []
    best_val_iou = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_metrics = evaluate(model, val_loader, criterion, device, threshold=0.5)
        elapsed = time.time() - t0

        current_lr = scheduler.get_last_lr()[0]

        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": val_metrics["loss"],
            "val_iou_fault": val_metrics["iou_fault"],
            "val_tol_iou": val_metrics["tol_iou"],
            "val_f1": val_metrics["f1"],
            "val_miou": val_metrics["miou"],
            "lr_encoder": current_lr,
            "elapsed_s": round(elapsed, 1),
        }
        train_log.append(entry)

        improved = ""
        if val_metrics["iou_fault"] > best_val_iou:
            best_val_iou = val_metrics["iou_fault"]
            patience_counter = 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            improved = " *"
        else:
            patience_counter += 1

        print(f"{epoch:5d}  {train_loss:10.6f}  {val_metrics['loss']:10.6f}  "
              f"{val_metrics['iou_fault']:9.4f}  {val_metrics['tol_iou']:9.4f}  "
              f"{val_metrics['f1']:7.4f}  {current_lr:10.2e}{improved}")

        with open(run_dir / "train_log.json", "w") as f:
            json.dump(train_log, f, indent=2)

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
            break

    # ── Threshold sweep on val ──
    print(f"\nBest val IoU_fault: {best_val_iou:.4f}")
    print("\nLoading best model for threshold sweep...")
    model = build_model(num_channels=args.channels, num_labels=2, dropout=args.dropout).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))

    thresholds = [round(0.05 * i, 2) for i in range(1, 11)]
    best_pixel_thresh, best_tol_thresh, sweep_results = threshold_sweep(
        model, val_loader, device, thresholds)
    print(f"\nThreshold sweep (val):  pIoU=pixel IoU, tIoU=tolerance IoU (3px dilation)")
    print(f"{'Thresh':>8s}  {'pIoU':>7s}  {'tIoU':>7s}  {'F1':>7s}  {'Prec':>7s}  {'Rec':>7s}")
    print("-" * 55)
    for t_str, m in sorted(sweep_results.items()):
        t_val = float(t_str)
        marker = ""
        if t_val == best_pixel_thresh: marker += " p*"
        if t_val == best_tol_thresh:   marker += " t*"
        print(f"{t_str:>8s}  {m['iou_fault']:7.4f}  {m['tol_iou']:7.4f}  "
              f"{m['f1']:7.4f}  {m['precision']:7.4f}  {m['recall']:7.4f}{marker}")
    print(f"\nBest threshold (pixel IoU): {best_pixel_thresh:.2f}")
    print(f"Best threshold (tol IoU):   {best_tol_thresh:.2f}")

    # ── Test evaluation ──
    # Score test set at the pixel-IoU-best threshold (this is the headline
    # number that stays comparable with prior runs). The returned dict also
    # carries the tolerance IoU at that same threshold for context.
    print(f"\nTest evaluation (threshold={best_pixel_thresh:.2f})...")
    test_metrics = evaluate(model, test_loader, criterion, device, threshold=best_pixel_thresh)
    print(f"  IoU_fault:  {test_metrics['iou_fault']:.4f}")
    print(f"  tol_iou:    {test_metrics['tol_iou']:.4f}")
    print(f"  F1:         {test_metrics['f1']:.4f}")
    print(f"  mIoU:       {test_metrics['miou']:.4f}")
    print(f"  Precision:  {test_metrics['precision']:.4f}")
    print(f"  Recall:     {test_metrics['recall']:.4f}")

    # Also score test at the tol-IoU-best threshold (separate report so the
    # two numbers are comparable, not mixed).
    if best_tol_thresh != best_pixel_thresh:
        print(f"\nTest evaluation (threshold={best_tol_thresh:.2f}, tol-IoU best)...")
        test_metrics_tol = evaluate(model, test_loader, criterion, device,
                                    threshold=best_tol_thresh)
        print(f"  IoU_fault:  {test_metrics_tol['iou_fault']:.4f}")
        print(f"  tol_iou:    {test_metrics_tol['tol_iou']:.4f}")
        print(f"  F1:         {test_metrics_tol['f1']:.4f}")
    else:
        test_metrics_tol = test_metrics

    # ── Save final metrics ──
    final = {
        "best_val_iou_fault": best_val_iou,
        "best_threshold_pixel": best_pixel_thresh,
        "best_threshold_tol": best_tol_thresh,
        "tolerance_px": TOLERANCE_PX,
        "threshold_sweep": sweep_results,
        "test_at_pixel_best": test_metrics,
        "test_at_tol_best": test_metrics_tol,
        "config": config,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(final, f, indent=2)

    print(f"\nAll results saved to: {run_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
