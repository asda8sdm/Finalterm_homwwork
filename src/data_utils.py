from __future__ import annotations

import math
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
UCI_ZIP = PROJECT_DIR / "individual+household+electric+power+consumption.zip"
WEATHER_GZ = RAW_DIR / "MENSQ_92_previous-1950-2024.csv.gz"
WEATHER_URL = (
    "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/"
    "BASE/MENS/MENSQ_92_previous-1950-2024.csv.gz"
)

TARGET_COL = "global_active_power"
POWER_SUM_COLS = [
    "global_active_power",
    "global_reactive_power",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
]
POWER_MEAN_COLS = ["voltage", "global_intensity"]
WEATHER_COLS = ["RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]
CALENDAR_COLS = [
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "dayofyear_sin",
    "dayofyear_cos",
]
FEATURE_COLS = (
    POWER_SUM_COLS
    + POWER_MEAN_COLS
    + ["sub_metering_remainder"]
    + WEATHER_COLS
    + CALENDAR_COLS
)


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_weather_if_needed() -> None:
    ensure_dirs()
    root_copy = PROJECT_DIR / WEATHER_GZ.name
    if WEATHER_GZ.exists():
        return
    if root_copy.exists() and root_copy.stat().st_size > 0:
        root_copy.replace(WEATHER_GZ)
        return
    urllib.request.urlretrieve(WEATHER_URL, WEATHER_GZ)


def _read_power_data() -> pd.DataFrame:
    df = pd.read_csv(
        UCI_ZIP,
        sep=";",
        compression="zip",
        na_values="?",
        low_memory=False,
    )
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    rename = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename)
    numeric_cols = POWER_SUM_COLS + POWER_MEAN_COLS
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    df = df.set_index("datetime")[numeric_cols]
    df = df.interpolate(method="time", limit_direction="both")
    daily = df.resample("D").agg({**{c: "sum" for c in POWER_SUM_COLS}, **{c: "mean" for c in POWER_MEAN_COLS}})
    daily = daily.asfreq("D")
    daily = daily.interpolate(method="time", limit_direction="both")
    daily["sub_metering_remainder"] = (
        daily["global_active_power"] * 1000.0 / 60.0
        - daily[["sub_metering_1", "sub_metering_2", "sub_metering_3"]].sum(axis=1)
    )
    daily["sub_metering_remainder"] = daily["sub_metering_remainder"].clip(lower=0)
    daily.index.name = "date"
    return daily.reset_index()


def _read_weather_data() -> pd.DataFrame:
    download_weather_if_needed()
    weather = pd.read_csv(WEATHER_GZ, sep=";", compression="gzip", low_memory=False)
    weather.columns = [c.strip() for c in weather.columns]
    if "AAAAMM" not in weather.columns:
        raise ValueError(f"Weather file does not contain AAAAMM; columns={weather.columns.tolist()[:20]}")
    for col in WEATHER_COLS:
        if col not in weather.columns:
            weather[col] = np.nan
        weather[col] = pd.to_numeric(weather[col], errors="coerce")

    weather["month"] = pd.to_datetime(weather["AAAAMM"].astype(str), format="%Y%m", errors="coerce")
    weather = weather.dropna(subset=["month"])

    station_col = "NUM_POSTE" if "NUM_POSTE" in weather.columns else None
    if station_col:
        needed = weather[weather["month"].between("2006-12-01", "2010-11-01")].copy()
        scores = needed.groupby(station_col)[WEATHER_COLS].apply(lambda x: x.notna().mean().mean())
        station = scores.sort_values(ascending=False).index[0]
        weather = weather[weather[station_col] == station].copy()

    weather = weather.sort_values("month").groupby("month", as_index=False)[WEATHER_COLS].mean()
    weather["RR"] = weather["RR"] / 10.0
    weather[WEATHER_COLS] = weather[WEATHER_COLS].interpolate(limit_direction="both").fillna(0.0)
    return weather


def build_daily_dataset(force: bool = False) -> pd.DataFrame:
    ensure_dirs()
    out = PROCESSED_DIR / "daily_power_weather.csv"
    train_out = PROCESSED_DIR / "train.csv"
    test_out = PROCESSED_DIR / "test.csv"
    if out.exists() and train_out.exists() and test_out.exists() and not force:
        return pd.read_csv(out, parse_dates=["date"])

    daily = _read_power_data()
    weather = _read_weather_data()
    daily["month"] = daily["date"].values.astype("datetime64[M]")
    merged = daily.merge(weather, on="month", how="left")
    merged[WEATHER_COLS] = merged[WEATHER_COLS].interpolate(limit_direction="both").fillna(0.0)

    dt = pd.to_datetime(merged["date"])
    merged["dow_sin"] = np.sin(2 * math.pi * dt.dt.dayofweek / 7)
    merged["dow_cos"] = np.cos(2 * math.pi * dt.dt.dayofweek / 7)
    merged["month_sin"] = np.sin(2 * math.pi * dt.dt.month / 12)
    merged["month_cos"] = np.cos(2 * math.pi * dt.dt.month / 12)
    merged["dayofyear_sin"] = np.sin(2 * math.pi * dt.dt.dayofyear / 365.25)
    merged["dayofyear_cos"] = np.cos(2 * math.pi * dt.dt.dayofyear / 365.25)

    merged = merged[["date"] + FEATURE_COLS].sort_values("date").reset_index(drop=True)
    split_idx = int(len(merged) * 0.65)
    merged["split"] = "train"
    merged.loc[split_idx:, "split"] = "test"
    merged.to_csv(out, index=False)
    merged.loc[merged["split"] == "train"].to_csv(train_out, index=False)
    merged.loc[merged["split"] == "test"].to_csv(test_out, index=False)
    return merged


class WindowedSeries:
    def __init__(
        self,
        frame: pd.DataFrame,
        input_len: int,
        horizon: int,
        feature_scaler: StandardScaler | None = None,
        target_scaler: StandardScaler | None = None,
        fit: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.input_len = input_len
        self.horizon = horizon
        self.feature_cols = FEATURE_COLS
        self.target_col = TARGET_COL

        features = self.frame[self.feature_cols].to_numpy(dtype=np.float32)
        target = self.frame[[self.target_col]].to_numpy(dtype=np.float32)
        if fit:
            self.feature_scaler = StandardScaler().fit(features)
            self.target_scaler = StandardScaler().fit(target)
        else:
            if feature_scaler is None or target_scaler is None:
                raise ValueError("feature_scaler and target_scaler are required when fit=False")
            self.feature_scaler = feature_scaler
            self.target_scaler = target_scaler

        self.features = self.feature_scaler.transform(features).astype(np.float32)
        self.target_scaled = self.target_scaler.transform(target).astype(np.float32).reshape(-1)
        self.target_raw = target.reshape(-1)
        self.dates = pd.to_datetime(self.frame["date"]).to_numpy()
        self.starts = np.arange(0, len(self.frame) - input_len - horizon + 1)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        start = int(self.starts[idx])
        x = self.features[start : start + self.input_len]
        y = self.target_scaled[start + self.input_len : start + self.input_len + self.horizon]
        return x, y

    def raw_target_for(self, idx: int) -> np.ndarray:
        start = int(self.starts[idx])
        return self.target_raw[start + self.input_len : start + self.input_len + self.horizon]

    def dates_for(self, idx: int) -> np.ndarray:
        start = int(self.starts[idx])
        return self.dates[start + self.input_len : start + self.input_len + self.horizon]

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        shape = values.shape
        flat = values.reshape(-1, 1)
        return self.target_scaler.inverse_transform(flat).reshape(shape)


def make_train_test(input_len: int, horizon: int) -> tuple[WindowedSeries, WindowedSeries]:
    frame = build_daily_dataset()
    train = frame[frame["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    test = frame[frame["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)
    train_ds = WindowedSeries(train, input_len, horizon, fit=True)
    test_ds = WindowedSeries(
        test,
        input_len,
        horizon,
        feature_scaler=train_ds.feature_scaler,
        target_scaler=train_ds.target_scaler,
    )
    if len(test_ds) <= 0:
        raise ValueError(
            f"Test split is too short for input_len={input_len}, horizon={horizon}; "
            f"test days={len(test)}"
        )
    return train_ds, test_ds
