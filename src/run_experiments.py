from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from data_utils import FEATURE_COLS, TARGET_COL, WEATHER_COLS, build_daily_dataset, make_train_test
from models import build_model


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def collate(batch: list[tuple[np.ndarray, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    return torch.tensor(np.stack(xs), dtype=torch.float32), torch.tensor(np.stack(ys), dtype=torch.float32)


def split_train_val(n: int, val_ratio: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    val_size = max(1, int(n * val_ratio))
    train_idx = np.arange(0, n - val_size)
    val_idx = np.arange(n - val_size, n)
    return train_idx, val_idx


def train_one(
    model_name: str,
    horizon: int,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
) -> tuple[dict[str, float | int | str], dict[str, np.ndarray]]:
    set_seed(seed)
    train_ds, test_ds = make_train_test(input_len=90, horizon=horizon)
    weather_indices = [FEATURE_COLS.index(c) for c in WEATHER_COLS]
    model = build_model(model_name, input_dim=len(FEATURE_COLS), horizon=horizon, weather_indices=weather_indices).to(device)

    train_idx, val_idx = split_train_val(len(train_ds))
    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        Subset(train_ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                val_losses.append(loss_fn(model(xb), yb).item())
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[{model_name} h={horizon} seed={seed}] "
                f"epoch={epoch:03d} train={np.mean(train_losses):.5f} val={val_loss:.5f}",
                flush=True,
            )
        if bad_epochs >= patience:
            print(f"[{model_name} h={horizon} seed={seed}] early stop at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    pred_scaled_batches = []
    true_scaled_batches = []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred_scaled_batches.append(model(xb.to(device)).cpu().numpy())
            true_scaled_batches.append(yb.numpy())
    pred_scaled = np.concatenate(pred_scaled_batches, axis=0)
    true_scaled = np.concatenate(true_scaled_batches, axis=0)
    pred = test_ds.inverse_target(pred_scaled)
    true = test_ds.inverse_target(true_scaled)
    mse = float(np.mean((pred - true) ** 2))
    mae = float(np.mean(np.abs(pred - true)))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "horizon": horizon,
            "seed": seed,
            "state_dict": model.state_dict(),
            "feature_cols": FEATURE_COLS,
            "target_col": TARGET_COL,
        },
        MODEL_DIR / f"{model_name}_h{horizon}_seed{seed}.pt",
    )

    plot_idx = len(test_ds) - 1
    plot_x = {
        "dates": test_ds.dates_for(plot_idx),
        "truth": test_ds.raw_target_for(plot_idx),
        "pred": pred[plot_idx],
    }
    row = {
        "model": model_name,
        "horizon": horizon,
        "seed": seed,
        "mse": mse,
        "mae": mae,
        "best_val_scaled_mse": float(best_val),
        "train_windows": len(train_idx),
        "val_windows": len(val_idx),
        "test_windows": len(test_ds),
    }
    print(f"[{model_name} h={horizon} seed={seed}] test mse={mse:.3f} mae={mae:.3f}", flush=True)
    return row, plot_x


def save_summary(rows: list[dict[str, float | int | str]]) -> pd.DataFrame:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(TABLE_DIR / "all_runs.csv", index=False)
    summary = (
        raw.groupby(["horizon", "model"], as_index=False)
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            best_val_scaled_mse_mean=("best_val_scaled_mse", "mean"),
        )
        .sort_values(["horizon", "mse_mean"])
    )
    summary.to_csv(TABLE_DIR / "summary_metrics.csv", index=False)
    md = ["| horizon | model | mse_mean | mse_std | mae_mean | mae_std |", "|---:|---|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        md.append(
            f"| {row.horizon} | {row.model} | {row.mse_mean:.4f} | {row.mse_std:.4f} | "
            f"{row.mae_mean:.4f} | {row.mae_std:.4f} |"
        )
    (TABLE_DIR / "summary_metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


def save_plots(plot_cache: dict[int, dict[str, dict[str, np.ndarray]]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for horizon, by_model in plot_cache.items():
        if not by_model:
            continue
        first = next(iter(by_model.values()))
        dates = pd.to_datetime(first["dates"])
        plt.figure(figsize=(12, 5))
        plt.plot(dates, first["truth"], color="black", linewidth=2.0, label="Ground Truth")
        styles = {
            "lstm": ("#4C78A8", "--"),
            "transformer": ("#F58518", "-."),
            "msrt": ("#54A24B", "-"),
        }
        for model_name, payload in by_model.items():
            color, linestyle = styles.get(model_name, (None, "-"))
            plt.plot(dates, payload["pred"], linewidth=1.6, linestyle=linestyle, color=color, label=model_name)
        plt.title(f"Power Forecast Comparison, Horizon={horizon} days")
        plt.xlabel("Date")
        plt.ylabel("Daily global active power")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"comparison_h{horizon}.png", dpi=180)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["lstm", "transformer", "msrt"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[90, 365])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028, 2029, 2030])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-preprocess", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.", flush=True)
    print(f"Using device: {device}", flush=True)
    daily = build_daily_dataset(force=args.force_preprocess)
    metadata = {
        "n_days": int(len(daily)),
        "date_start": str(pd.to_datetime(daily["date"]).min().date()),
        "date_end": str(pd.to_datetime(daily["date"]).max().date()),
        "train_days": int((daily["split"] == "train").sum()),
        "test_days": int((daily["split"] == "test").sum()),
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)

    rows: list[dict[str, float | int | str]] = []
    plot_cache: dict[int, dict[str, dict[str, np.ndarray]]] = {h: {} for h in args.horizons}
    first_seed = args.seeds[0]
    for horizon in args.horizons:
        for model_name in args.models:
            for seed in args.seeds:
                row, plot_payload = train_one(
                    model_name=model_name,
                    horizon=horizon,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    patience=args.patience,
                    device=device,
                )
                rows.append(row)
                if seed == first_seed:
                    plot_cache[horizon][model_name] = plot_payload
                save_summary(rows)
                save_plots(plot_cache)
    summary = save_summary(rows)
    save_plots(plot_cache)
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
