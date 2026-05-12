import gc, json, time, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F
import albumentations as A

# ── Paths ─────────────────────────────────────────────────
BASE_DIR   = Path(os.environ.get(
    'FAULT_DATA_DIR',
    Path.home() / 'Desktop' / 'prithvi' / 'data'
))
PATCH_BASE = BASE_DIR / 'patches'
CKPT_BASE  = BASE_DIR / 'checkpoints'
CKPT_BASE.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU:  {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

# 학습할 지역 순서 — fault 비율 높은 것부터 (성공 가능성 높음)
REGIONS_TO_TRAIN = [
    'imperial_valley',   # fault 65%
    'owens_valley',      # fault 68%
    'mojave',            # fault 57%
    'pauma_valley',      # fault 53%
    'sierra_pelona',     # fault 8% (재학습)
]
PATCH_SIZE_STR = '256'


# ── Augmentation ──────────────────────────────────────────
def get_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1,
            rotate_limit=30, border_mode=0, p=0.4
        ),
        A.ElasticTransform(alpha=20, sigma=5, border_mode=0, p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.4, contrast_limit=0.3, p=0.5
        ),
        A.GaussNoise(var_limit=(5.0, 20.0), mean=0, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ], additional_targets={'mask': 'mask'})


# ── Dataset ───────────────────────────────────────────────
class DEMDataset(Dataset):
    def __init__(self, npz_path, augment=False):
        data  = np.load(npz_path)
        imgs  = np.clip(np.nan_to_num(
            data['images'].astype(np.float32), nan=0.0), 0, 1)
        masks = data['masks'].astype(np.int64)

        self.images    = imgs
        self.masks     = masks
        self.augment   = augment
        self.aug_fn    = get_augmentation() if augment else None
        self.has_fault = (masks.sum(axis=(1, 2)) > 0)

        n, nf = len(self.images), int(self.has_fault.sum())
        print(f'    {Path(npz_path).name}: {n} | fault={nf} ({nf/n*100:.0f}%)')

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img  = self.images[idx].copy()
        mask = self.masks[idx].copy()

        if self.augment and self.aug_fn is not None:
            img_hwc  = img.transpose(1, 2, 0)
            mask_f32 = mask.astype(np.float32)
            result   = self.aug_fn(image=img_hwc, mask=mask_f32)
            img      = result['image'].transpose(2, 0, 1)
            mask     = result['mask'].astype(np.int64)
            img      = np.clip(img, 0, 1)

        return torch.from_numpy(img).float(), torch.from_numpy(mask)


# ── Loss ──────────────────────────────────────────────────
class FaultLoss(nn.Module):
    def __init__(self, fault_weight=5.0):
        super().__init__()
        w       = torch.tensor([1.0, fault_weight]).float().to(device)
        self.ce = nn.CrossEntropyLoss(weight=w)

    def forward(self, logits, targets):
        ce    = self.ce(logits, targets)
        prob  = torch.softmax(logits, dim=1)[:, 1]
        tgt_f = (targets == 1).float()
        inter = (prob * tgt_f).sum(dim=(1, 2))
        union = prob.sum(dim=(1, 2)) + tgt_f.sum(dim=(1, 2))
        dice  = 1.0 - (2*inter + 1) / (union + 1)
        return ce + dice.mean()


# ── Metrics ───────────────────────────────────────────────
def compute_metrics(logits, targets, threshold=0.5):
    prob  = torch.softmax(logits, dim=1)[:, 1]
    preds = (prob > threshold).long()
    tp = ((preds==1)&(targets==1)).sum().float()
    fp = ((preds==1)&(targets==0)).sum().float()
    fn = ((preds==0)&(targets==1)).sum().float()
    tn = ((preds==0)&(targets==0)).sum().float()
    return {
        'mIoU':      ((tp/(tp+fp+fn+1e-8)+tn/(tn+fp+fn+1e-8))/2).item(),
        'IoU_fault': (tp/(tp+fp+fn+1e-8)).item(),
        'F1':        (2*tp/(2*tp+fp+fn+1e-8)).item(),
    }


def tune_threshold(model, val_ds, batch_size=16):
    loader = DataLoader(val_ds, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    model.eval()
    all_probs, all_masks = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs   = imgs.float().to(device)
            logits = forward_segformer(model, imgs)
            all_probs.append(logits.softmax(1)[:, 1].cpu().numpy())
            all_masks.append(masks.numpy())
    probs = np.concatenate(all_probs).ravel()
    masks = np.concatenate(all_masks).ravel()

    best_t, best_iou = 0.5, 0.0
    for t in np.arange(0.10, 0.51, 0.02):
        pred = (probs > t).astype(int)
        tp   = ((pred==1)&(masks==1)).sum()
        fp   = ((pred==1)&(masks==0)).sum()
        fn   = ((pred==0)&(masks==1)).sum()
        iou  = tp/(tp+fp+fn+1e-8)
        if iou > best_iou:
            best_iou, best_t = iou, t
    print(f'Best threshold: {best_t:.2f} (IoU={best_iou:.4f})')
    return best_t, best_iou


def evaluate_test(model, test_ds, threshold, batch_size=16):
    loader = DataLoader(test_ds, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    model.eval()
    all_probs, all_masks = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs   = imgs.float().to(device)
            logits = forward_segformer(model, imgs)
            all_probs.append(logits.softmax(1)[:, 1].cpu().numpy())
            all_masks.append(masks.numpy())
    probs = np.concatenate(all_probs).ravel()
    masks = np.concatenate(all_masks).ravel()
    pred  = (probs > threshold).astype(int)
    tp = ((pred==1)&(masks==1)).sum()
    fp = ((pred==1)&(masks==0)).sum()
    fn = ((pred==0)&(masks==1)).sum()
    tn = ((pred==0)&(masks==0)).sum()
    return {
        'threshold': float(threshold),
        'mIoU':      float((tp/(tp+fp+fn+1e-8)+tn/(tn+fp+fn+1e-8))/2),
        'IoU_fault': float(tp/(tp+fp+fn+1e-8)),
        'F1':        float(2*tp/(2*tp+fp+fn+1e-8)),
        'Accuracy':  float((tp+tn)/(tp+tn+fp+fn+1e-8)),
    }


# ── SegFormer-B2 ──────────────────────────────────────────
def build_segformer_b2(num_labels=2):
    model = SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/mit-b2',
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        id2label={0: 'background', 1: 'fault'},
        label2id={'background': 0, 'fault': 1},
    )
    return model.to(device)


def forward_segformer(model, imgs):
    H, W = imgs.shape[-2], imgs.shape[-1]
    out  = model(pixel_values=imgs).logits
    return F.interpolate(out, size=(H, W),
                         mode='bilinear', align_corners=False)


def get_optimizer(model, encoder_lr=1e-5, decoder_lr=1e-4,
                  weight_decay=1e-4):
    encoder_params = list(model.segformer.parameters())
    decoder_params = list(model.decode_head.parameters())
    return optim.AdamW([
        {'params': encoder_params, 'lr': encoder_lr},
        {'params': decoder_params, 'lr': decoder_lr},
    ], weight_decay=weight_decay)


# ── Training (단일 지역) ──────────────────────────────────
def train_region(region, num_epochs=80, patience=15, batch_size=16,
                 fault_weight=5.0, oversample_w=50.0,
                 encoder_lr=1e-5, decoder_lr=1e-4):

    patch_dir = PATCH_BASE / f'{region}_dem_{PATCH_SIZE_STR}'
    ckpt_dir  = CKPT_BASE  / f'segformer_b2_{region}_{PATCH_SIZE_STR}'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"#"*60}')
    print(f'# REGION: {region.upper()}')
    print(f'{"#"*60}')

    print('\n[Train]')
    train_ds = DEMDataset(patch_dir/'train.npz', augment=True)
    print('[Val]')
    val_ds   = DEMDataset(patch_dir/'val.npz', augment=False)
    print('[Test]')
    test_ds  = DEMDataset(patch_dir/'test.npz', augment=False)

    weights = np.where(train_ds.has_fault, oversample_w, 1.0)
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights).float(),
        num_samples=len(weights), replacement=True
    )
    LOADER_KW    = dict(num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, **LOADER_KW)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, **LOADER_KW)

    model     = build_segformer_b2()
    criterion = FaultLoss(fault_weight=fault_weight)
    optimizer = get_optimizer(model, encoder_lr, decoder_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-7)
    scaler    = torch.amp.GradScaler('cuda')

    best_iou, patience_cnt = 0.0, 0
    history = {'epoch':[], 'train_loss':[], 'val_iou_fault':[], 'val_f1':[]}
    t0 = time.time()

    print(f'\n{"Ep":>4} | {"Loss":>8} | {"mIoU":>7} | '
          f'{"IoU_fault":>10} | {"F1":>7} | {"min":>5}')
    print('-'*55)

    for epoch in range(1, num_epochs+1):
        model.train()
        t_loss = 0.0
        for imgs, masks in train_loader:
            imgs  = imgs.float().to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                logits = forward_segformer(model, imgs)
                loss   = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            t_loss += loss.item()
        t_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        vm = {'mIoU':0.0, 'IoU_fault':0.0, 'F1':0.0}
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs  = imgs.float().to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    logits = forward_segformer(model, imgs)
                m = compute_metrics(logits, masks)
                for k in vm: vm[k] += m[k]
        for k in vm: vm[k] /= len(val_loader)

        history['epoch'].append(epoch)
        history['train_loss'].append(t_loss)
        history['val_iou_fault'].append(vm['IoU_fault'])
        history['val_f1'].append(vm['F1'])

        if epoch % 5 == 0 or epoch <= 3:
            elapsed = (time.time()-t0)/60
            print(f'{epoch:>4} | {t_loss:>8.4f} | {vm["mIoU"]:>7.4f} | '
                  f'{vm["IoU_fault"]:>10.4f} | {vm["F1"]:>7.4f} | '
                  f'{elapsed:>4.1f}m')

        if vm['IoU_fault'] > best_iou:
            best_iou     = vm['IoU_fault']
            patience_cnt = 0
            torch.save({'epoch': epoch,
                        'model_state': model.state_dict(),
                        'val_metrics': vm,
                        'region': region},
                       ckpt_dir/'best.pth')
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

    elapsed = (time.time()-t0)/60
    print(f'\nDone {region}: {elapsed:.1f}min | Best val IoU: {best_iou:.4f}')

    print('\n--- Threshold Tuning ---')
    ckpt = torch.load(ckpt_dir/'best.pth', map_location=device)
    model.load_state_dict(ckpt['model_state'])
    best_t, _ = tune_threshold(model, val_ds)

    print('\n--- Test Evaluation ---')
    results = evaluate_test(model, test_ds, best_t)
    print(f'  TEST IoU_fault: {results["IoU_fault"]:.4f}')
    print(f'  TEST F1:        {results["F1"]:.4f}')

    results.update({
        'region':       region,
        'patch_size':   PATCH_SIZE_STR,
        'model':        'SegFormer-B2',
        'best_epoch':   int(ckpt['epoch']),
        'val_iou':      float(best_iou),
        'training_min': float(elapsed),
    })

    with open(ckpt_dir/'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], 'b-')
    axes[0].set_title(f'B2 {region} — Train Loss')
    axes[0].set_xlabel('Epoch'); axes[0].grid(alpha=0.3)
    axes[1].plot(history['val_iou_fault'], 'r--', label='IoU_fault')
    axes[1].plot(history['val_f1'], 'b:', label='F1')
    axes[1].axhline(best_iou, color='r', ls=':', alpha=0.5,
                    label=f'Best:{best_iou:.4f}')
    axes[1].set_title(f'B2 {region} — Val Metrics')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ckpt_dir/f'training_curve.png', dpi=150)
    plt.close()

    del model, optimizer, scheduler, scaler, criterion
    del train_ds, val_ds, test_ds, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    return results


# ── Main: 모든 지역 순차 학습 ─────────────────────────────
if __name__ == '__main__':
    print(f'\n{"="*60}')
    print(f'Per-Region B2 Training')
    print(f'Regions: {", ".join(REGIONS_TO_TRAIN)}')
    print(f'{"="*60}\n')

    all_results = {}
    overall_t0  = time.time()

    for i, region in enumerate(REGIONS_TO_TRAIN):
        print(f'\n\n[{i+1}/{len(REGIONS_TO_TRAIN)}] Starting {region}...')
        try:
            results = train_region(
                region       = region,
                num_epochs   = 80,
                patience     = 15,
                batch_size   = 16,
                fault_weight = 5.0,
                oversample_w = 50.0,
                encoder_lr   = 1e-5,
                decoder_lr   = 1e-4,
            )
            all_results[region] = results
        except Exception as e:
            print(f'❌ {region} failed: {e}')
            all_results[region] = {'error': str(e)}

    # 최종 요약
    total_time = (time.time() - overall_t0) / 60
    print(f'\n\n{"="*60}')
    print(f'ALL REGIONS COMPLETE — Total: {total_time:.1f} min')
    print(f'{"="*60}')
    print(f'{"Region":<20} {"IoU_fault":>10} {"F1":>10} {"Best Epoch":>12}')
    print('-'*60)

    summary_path = CKPT_BASE / 'per_region_summary.json'
    summary = {'total_time_min': total_time, 'results': {}}

    for region, res in all_results.items():
        if 'error' in res:
            print(f'{region:<20} {"ERROR":>10}')
            summary['results'][region] = res
        else:
            print(f'{region:<20} {res["IoU_fault"]:>10.4f} '
                  f'{res["F1"]:>10.4f} {res["best_epoch"]:>12}')
            summary['results'][region] = {
                'IoU_fault': res['IoU_fault'],
                'F1':        res['F1'],
                'mIoU':      res['mIoU'],
                'threshold': res['threshold'],
                'best_epoch': res['best_epoch'],
            }

    print('-'*60)
    print(f'B2 Carrizo (existing baseline): IoU_fault = 0.3687')

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSummary saved: {summary_path}')
