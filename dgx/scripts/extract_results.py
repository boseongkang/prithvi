"""Extract config + metrics from all runs. Region inferred from train_size."""
import json
from pathlib import Path

# train_size -> region mapping (from today's work)
SIZE_TO_REGION = {
    3479: "big_pine",
    13921: "sierra_nevada",
    9656: "death_valley",
    4977: "panamint",
}

runs = sorted(Path("runs").glob("*/"))
print(f"{'run':<17} {'region':<14} {'ch':<3} {'patch':<6} {'val_iou':<8} {'test_iou':<9} {'recall':<8} {'prec':<8} {'f1':<8} {'train_n':<8}")
print("-" * 105)
for d in runs:
    cfg_f, met_f = d / "config.json", d / "metrics.json"
    if not cfg_f.exists():
        continue
    cfg = json.loads(cfg_f.read_text())
    met = json.loads(met_f.read_text()) if met_f.exists() else {}

    ch = cfg.get("channels", "?")
    train_n = cfg.get("train_size", "?")
    region = SIZE_TO_REGION.get(train_n, f"?({train_n})")
    val_iou = met.get("best_val_iou_fault", "?")
    test = met.get("test_at_pixel_best", {})
    test_iou = test.get("iou_fault", "?")
    recall = test.get("recall", "?")
    prec = test.get("precision", "?")
    f1 = test.get("f1", "?")
    patch = cfg.get("patch_size", 256)  # default 256

    print(f"{d.name:<17} {str(region):<14} {str(ch):<3} {str(patch):<6} "
          f"{str(val_iou):<8} {str(test_iou):<9} {str(recall):<8} {str(prec):<8} {str(f1):<8} {str(train_n):<8}")
