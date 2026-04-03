"""Train a bus delay prediction model.

Script loads `bus_delay.csv`, preprocesses data, trains a RandomForestRegressor,
prints evaluation metrics, and saves the model artifacts (model, scaler, label encoder).

Usage:
    python train_model.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the dataset from CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """Preprocess the raw dataset and return features, target, and fitted LabelEncoder."""

    # Drop columns that are not needed for prediction
    cols_to_drop = [
        "bus_id",
        "origin_station",
        "destination_station",
        "scheduled_departure",
        "scheduled_arrival",
        "date",
        "delayed",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # Fill missing values
    if "weather_condition" in df.columns:
        weather_mode = df["weather_condition"].mode(dropna=True)
        if not weather_mode.empty:
            df["weather_condition"] = df["weather_condition"].fillna(weather_mode.iloc[0])
        else:
            df["weather_condition"] = df["weather_condition"].fillna("Unknown")

    if "holiday" in df.columns:
        df["holiday"] = df["holiday"].fillna(0)

    # Drop rows missing the target
    df = df.dropna(subset=["actual_arrival_delay_min"])

    # Encode categorical variables
    label_encoder = LabelEncoder()
    if "weather_condition" in df.columns:
        df["weather_condition_encoded"] = label_encoder.fit_transform(df["weather_condition"].astype(str))
        df = df.drop(columns=["weather_condition"])

    # Separate features/target
    X = df.drop(columns=["actual_arrival_delay_min"])  # type: ignore
    y = df["actual_arrival_delay_min"]

    return X, y, label_encoder


def build_pipeline(X: pd.DataFrame) -> StandardScaler:
    """Fit and return a StandardScaler for the input features."""
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def train_model(X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    """Train a RandomForestRegressor and return the fitted model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


def evaluate_model(model: RandomForestRegressor, X_test: np.ndarray, y_test: np.ndarray) -> tuple[float, float]:
    """Evaluate the model and return MAE and R^2 score."""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return mae, r2


def save_artifacts(model: RandomForestRegressor, scaler: StandardScaler, label_encoder: LabelEncoder, out_dir: Path) -> None:
    """Save model artifacts to the output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.pkl")
    joblib.dump(scaler, out_dir / "scaler.pkl")
    joblib.dump(label_encoder, out_dir / "label_encoder.pkl")
    logging.info("Saved model artifacts to %s", out_dir)


def main() -> None:
    """Main entrypoint for training the model."""
    try:
        base_dir = Path(__file__).resolve().parent
        data_path = base_dir / "bus_delay.csv"
        logging.info("Loading data from %s", data_path)
        df = load_data(data_path)

        logging.info("Preprocessing data")
        X, y, label_encoder = preprocess(df)

        logging.info("Scaling features")
        scaler = build_pipeline(X)
        X_scaled = scaler.transform(X)

        logging.info("Splitting into train/test")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y.values, test_size=0.2, random_state=42
        )

        logging.info("Training RandomForestRegressor")
        model = train_model(X_train, y_train)

        logging.info("Evaluating model")
        mae, r2 = evaluate_model(model, X_test, y_test)
        logging.info("Model evaluation: MAE=%.2f, R^2=%.3f", mae, r2)

        save_artifacts(model, scaler, label_encoder, base_dir)

        print(f"Training complete. MAE: {mae:.2f}, R^2: {r2:.3f}")

    except Exception as exc:  # pragma: no cover
        logging.exception("Failed to train model")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
