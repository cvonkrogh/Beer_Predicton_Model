import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROCESSED_DATA_PATH = "data/processed/weekly_beer_data.csv"
LAGS = [1, 2, 3, 4, 8, 12, 26, 52]
TEST_WEEKS = 12


# ==============================
# CREATE LAGS
# ==============================

def create_lags(df):
    df = df.sort_values("week")
    for lag in LAGS:
        df[f"lag_{lag}"] = df["liters"].shift(lag)
    return df


# ==============================
# SMAPE FUNCTION
# ==============================

def smape(y_true, y_pred):
    return np.mean(
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred) + 1e-6)
    ) * 100


# ==============================
# EVALUATE SINGLE BEER
# ==============================

def evaluate_beer(beer_df, beer_name):

    beer_df = create_lags(beer_df)
    beer_df = beer_df.dropna().copy()

    beer_df["liters"] = beer_df["liters"].clip(lower=0)

    feature_cols = [f"lag_{lag}" for lag in LAGS] + [
        "week_of_year",
        "month",
        "year",
        "hot_week",
        "rainy_week"
    ]

    train = beer_df.iloc[:-TEST_WEEKS]
    test = beer_df.iloc[-TEST_WEEKS:]

    X_train = train[feature_cols]
    y_train = np.log1p(train["liters"])

    X_test = test[feature_cols]
    y_test = test["liters"].values

    # -------------------------
    # Train Model
    # -------------------------
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)

    # -------------------------
    # Metrics
    # -------------------------
    model_mae = mean_absolute_error(y_test, preds)
    model_rmse = np.sqrt(mean_squared_error(y_test, preds))
    model_smape = smape(y_test, preds)

    # -------------------------
    # Naive Baseline (Last Week)
    # -------------------------
    naive_preds = test["lag_1"].values

    naive_mae = mean_absolute_error(y_test, naive_preds)
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_preds))
    naive_smape = smape(y_test, naive_preds)

    # -------------------------
    # Plot Actual vs Predicted
    # -------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(test["week"], y_test, label="Actual")
    plt.plot(test["week"], preds, label="Model Forecast")
    plt.plot(test["week"], naive_preds, label="Naive Forecast", linestyle="--")
    plt.title(f"{beer_name} – Last 12 Weeks Backtest")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return {
        "beer": beer_name,
        "Model MAE": model_mae,
        "Naive MAE": naive_mae,
        "Model RMSE": model_rmse,
        "Naive RMSE": naive_rmse,
        "Model SMAPE (%)": model_smape,
        "Naive SMAPE (%)": naive_smape
    }


# ==============================
# MAIN
# ==============================

def main():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["week"] = pd.to_datetime(df["week"])

    results = []

    for beer in df["beer"].unique():
        print(f"\nEvaluating {beer}...")
        beer_df = df[df["beer"] == beer].copy()
        result = evaluate_beer(beer_df, beer)
        results.append(result)

    results_df = pd.DataFrame(results)

    print("\n======================================")
    print("MODEL vs NAIVE BASELINE (Last 12 Weeks)")
    print("======================================\n")
    print(results_df)


if __name__ == "__main__":
    main()