# notebooks/evaluate_models.py

import os
import sqlite3

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

DATA_PATH = "data/Datamlops.xlsx"
TARGET_COL = "TGM"
YEAR_COL = "Year"
MODEL_DIR = "models"
ARIMA_PATH = os.path.join(MODEL_DIR, "arima_tgm.pkl")


def load_data_with_tgm():
    df = pd.read_excel(DATA_PATH)

    df = df.rename(
        columns={
            "Reading Frequency per week": "RF",
            "Number of Readings per Quarter": "NR",
            "Daily Reading Duration (in minutes)": "DRD",
            "Internet Access Frequency per Week": "IAF",
            "Daily Internet Duration (in minutes)": "DID",
        }
    )

    required_cols = ["Provinsi", "RF", "DRD", "NR", "IAF", "DID", YEAR_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom berikut belum ada di data: {missing}")

    df[TARGET_COL] = (
        0.3 * df["RF"]
        + 0.3 * df["DRD"]
        + 0.3 * df["NR"]
        + 0.05 * df["IAF"]
        + 0.05 * df["DID"]
    )

    trend_df = df.groupby(YEAR_COL)[TARGET_COL].mean().sort_index()
    return trend_df


def train_arima(series_values, order=(1, 1, 1)):
    model = ARIMA(series_values, order=order)
    model_fit = model.fit()
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_fit.save(ARIMA_PATH)
    return model_fit


def naive_forecast(train, h):
    # baseline: nilai terakhir di training diulang untuk semua step
    return np.repeat(train[-1], h)


def evaluate_models(test, arima_forecast, naive_forecast_values):
    results = {}

    results["ARIMA_MAE"] = mean_absolute_error(test, arima_forecast)
    results["ARIMA_MAPE"] = mean_absolute_percentage_error(test, arima_forecast)

    results["Naive_MAE"] = mean_absolute_error(test, naive_forecast_values)
    results["Naive_MAPE"] = mean_absolute_percentage_error(test, naive_forecast_values)

    return results


def main():
    trend_df = load_data_with_tgm()
    values = trend_df.values.astype(float)

    # gunakan split sederhana: 80% train, 20% test
    n = len(values)
    split_idx = int(0.8 * n)
    train_values = values[:split_idx]
    test_values = values[split_idx:]

    # train ARIMA di data train
    arima_model = train_arima(train_values, order=(1, 1, 1))

    # forecast length = panjang test
    h = len(test_values)
    arima_forecast = arima_model.forecast(steps=h)

    # baseline: naive
    naive_forecast_values = naive_forecast(train_values, h)

    results = evaluate_models(test_values, arima_forecast, naive_forecast_values)

    print("=== Evaluation Results ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # simpan ke CSV untuk laporan
    eval_df = pd.DataFrame([results])
    eval_df.to_csv("data/evaluation_results.csv", index=False)
    print("\nSaved metrics to data/evaluation_results.csv")


if __name__ == "__main__":
    main()
