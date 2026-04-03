"""Flask API for Bus Delay Prediction.

Routes:
- GET / => renders the HTML UI
- POST /predict => returns predicted delay in minutes

Dependencies:
- flask
- flask_cors
- joblib
- numpy
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = Flask(__name__, template_folder="templates")
CORS(app)

MODEL: "RandomForestRegressor" | None = None  # type: ignore
SCALER: "StandardScaler" | None = None  # type: ignore
LABEL_ENCODER: "LabelEncoder" | None = None  # type: ignore


def load_artifacts(work_dir: Path) -> None:
    """Load model artifacts from disk."""
    global MODEL, SCALER, LABEL_ENCODER

    model_path = work_dir / "model.pkl"
    scaler_path = work_dir / "scaler.pkl"
    encoder_path = work_dir / "label_encoder.pkl"

    MODEL = joblib.load(model_path)
    SCALER = joblib.load(scaler_path)
    LABEL_ENCODER = joblib.load(encoder_path)

    logging.info("Loaded model artifacts")


@app.route("/", methods=["GET"])
def home() -> str:
    """Render the main application UI."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_delay() -> tuple[dict, int]:
    """Predict delay given weather, holiday, and peak hour flags."""
    try:
        payload = request.get_json(force=True)
        if not payload:
            return {"error": "Invalid input payload"}, 400

        weather_condition = payload.get("weather_condition")
        holiday = payload.get("holiday")
        peak_hour = payload.get("peak_hour")

        if weather_condition is None or holiday is None or peak_hour is None:
            return {"error": "Missing required fields"}, 400

        # Input sanitization
        weather_condition = str(weather_condition).strip()
        holiday_val = int(holiday)
        peak_hour_val = int(peak_hour)

        if LABEL_ENCODER is None or SCALER is None or MODEL is None:
            return {"error": "Model not loaded"}, 500

        try:
            weather_encoded = LABEL_ENCODER.transform([weather_condition])[0]
        except Exception:
            return {
                "error": f"Unknown weather condition: '{weather_condition}'. Valid options are: {', '.join(LABEL_ENCODER.classes_)}"
            }, 400

        features = np.array([[weather_encoded, holiday_val, peak_hour_val]], dtype=float)
        scaled = SCALER.transform(features)
        prediction = MODEL.predict(scaled)
        predicted_delay = float(np.round(prediction[0], 2))

        return {"predicted_delay": predicted_delay}, 200

    except ValueError as exc:
        logging.exception("Invalid input")
        return {"error": str(exc)}, 400
    except Exception as exc:  # pragma: no cover
        logging.exception("Prediction failed")
        return {"error": "Internal server error"}, 500


def main() -> None:
    """Application entrypoint."""
    work_dir = Path(__file__).resolve().parent
    try:
        load_artifacts(work_dir)
    except Exception as exc:  # pragma: no cover
        logging.exception("Failed to load model artifacts")
        raise SystemExit(1) from exc

    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()
