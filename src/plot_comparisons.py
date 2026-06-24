from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from data_utils import FEATURE_COLS, WEATHER_COLS, make_train_test
from models import build_model


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "outputs" / "models"
FIG_DIR = PROJECT_DIR / "outputs" / "figures"


def plot_horizon(horizon: int, seed: int = 2026, device: str = "cpu") -> None:
    _, test_ds = make_train_test(input_len=90, horizon=horizon)
    weather_indices = [FEATURE_COLS.index(c) for c in WEATHER_COLS]
    idx = len(test_ds) - 1
    x_np, y_np = test_ds[idx]
    x = torch.tensor(x_np[None, ...], dtype=torch.float32, device=device)
    dates = pd.to_datetime(test_ds.dates_for(idx))
    truth = test_ds.inverse_target(y_np[None, ...])[0]

    preds = {}
    for name in ["lstm", "transformer", "msrt"]:
        ckpt_path = MODEL_DIR / f"{name}_h{horizon}_seed{seed}.pt"
        ckpt = torch.load(ckpt_path, map_location=device)
        model = build_model(name, input_dim=len(FEATURE_COLS), horizon=horizon, weather_indices=weather_indices)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        with torch.no_grad():
            pred_scaled = model(x).cpu().numpy()
        preds[name] = test_ds.inverse_target(pred_scaled)[0]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(dates, truth, color="black", linewidth=2.0, label="Ground Truth")
    styles = {
        "lstm": ("#4C78A8", "--"),
        "transformer": ("#F58518", "-."),
        "msrt": ("#54A24B", "-"),
    }
    for name, pred in preds.items():
        color, linestyle = styles[name]
        plt.plot(dates, pred, linewidth=1.6, linestyle=linestyle, color=color, label=name)
    plt.title(f"Power Forecast Comparison, Horizon={horizon} days")
    plt.xlabel("Date")
    plt.ylabel("Daily global active power")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"comparison_h{horizon}.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(3, 1, figsize=(12, 8.2), sharex=True)
    pretty = {"lstm": "LSTM", "transformer": "Transformer", "msrt": "MSRT (ours)"}
    for ax, name in zip(axes, ["lstm", "transformer", "msrt"]):
        color, linestyle = styles[name]
        ax.plot(dates, truth, color="#2F2F2F", linewidth=1.8, label="Ground Truth")
        ax.plot(dates, preds[name], color=color, linestyle=linestyle, linewidth=1.7, label=pretty[name])
        ax.set_title(pretty[name], loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("Power")
        ax.grid(True, alpha=0.22)
        ax.legend(loc="upper right", frameon=False, ncol=2)
    axes[-1].set_xlabel("Date")
    fig.suptitle(f"Ground Truth vs Individual Model Forecasts, Horizon={horizon} days", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG_DIR / f"comparison_h{horizon}_split.png", dpi=180)
    plt.close(fig)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for horizon in [90, 365]:
        plot_horizon(horizon, device=device)
    print(f"Wrote comparison plots to {FIG_DIR}")


if __name__ == "__main__":
    main()
