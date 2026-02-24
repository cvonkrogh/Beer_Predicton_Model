import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

PROCESSED_DATA_PATH = "data/processed/weekly_beer_data.csv"

LEAD_TIME = 12
SMALL_TANK = 2000
LARGE_TANK = 6000
TEST_WEEKS = 52
LAGS = [1, 2, 3, 4, 8, 12, 26, 52]


# ==============================
# CREATE LAGS
# ==============================

def create_lags(df):
    df = df.sort_values("week")
    for lag in LAGS:
        df[f"lag_{lag}"] = df["liters"].shift(lag)
    return df


# ==============================
# TRAIN MODEL
# ==============================

def train_model(train_df):

    feature_cols = [f"lag_{lag}" for lag in LAGS] + [
        "week_of_year",
        "month",
        "year",
        "hot_week",
        "rainy_week"
    ]

    X = train_df[feature_cols]
    y = np.log1p(train_df["liters"].clip(lower=0))

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X, y)

    return model, feature_cols


# ==============================
# SIMULATION PER BEER
# ==============================

def simulate_beer(beer_df, beer_name):

    beer_df = create_lags(beer_df)
    beer_df = beer_df.dropna().copy()

    train = beer_df.iloc[:-TEST_WEEKS]
    test = beer_df.iloc[-TEST_WEEKS:].copy()

    model, feature_cols = train_model(train)

    inventory_ml = 0
    inventory_naive = 0

    stockouts_ml = 0
    stockouts_naive = 0

    shortage_ml = 0
    shortage_naive = 0

    tanks_ml = 0
    tanks_naive = 0

    prod_schedule_ml = {}
    prod_schedule_naive = {}

    for i in range(len(test)):

        week = test.iloc[i]["week"]
        actual = test.iloc[i]["liters"]

        # ========================
        # PRODUCTION ARRIVAL
        # ========================
        if week in prod_schedule_ml:
            inventory_ml += prod_schedule_ml[week]

        if week in prod_schedule_naive:
            inventory_naive += prod_schedule_naive[week]

        # ========================
        # APPLY DEMAND
        # ========================
        inventory_ml -= actual
        inventory_naive -= actual

        if inventory_ml < 0:
            stockouts_ml += 1
            shortage_ml += abs(inventory_ml)
            inventory_ml = 0

        if inventory_naive < 0:
            stockouts_naive += 1
            shortage_naive += abs(inventory_naive)
            inventory_naive = 0

        # ========================
        # LOOK-AHEAD FORECAST (12W)
        # ========================
        lookahead_end = min(i + LEAD_TIME, len(test))

        # ----- ML cumulative forecast -----
        future_slice = test.iloc[i:lookahead_end]

        X_future = future_slice[feature_cols]
        preds_log = model.predict(X_future)
        preds_ml = np.expm1(preds_log)
        cumulative_ml = preds_ml.sum()

        # ----- Naive cumulative forecast -----
        preds_naive = future_slice["lag_1"].values
        cumulative_naive = preds_naive.sum()

        # ========================
        # BREW DECISION (CUMULATIVE)
        # ========================

        # ML
        if inventory_ml < cumulative_ml:

            needed = cumulative_ml - inventory_ml

            while needed > 0:
                if needed > LARGE_TANK:
                    brew = LARGE_TANK
                else:
                    brew = SMALL_TANK

                arrival_index = i + LEAD_TIME

                if arrival_index < len(test):
                    arrival_week = test.iloc[arrival_index]["week"]
                    prod_schedule_ml[arrival_week] = prod_schedule_ml.get(arrival_week, 0) + brew
                    tanks_ml += 1

                needed -= brew

        # Naive
        if inventory_naive < cumulative_naive:

            needed = cumulative_naive - inventory_naive

            while needed > 0:
                if needed > LARGE_TANK:
                    brew = LARGE_TANK
                else:
                    brew = SMALL_TANK

                arrival_index = i + LEAD_TIME

                if arrival_index < len(test):
                    arrival_week = test.iloc[arrival_index]["week"]
                    prod_schedule_naive[arrival_week] = prod_schedule_naive.get(arrival_week, 0) + brew
                    tanks_naive += 1

                needed -= brew

    return {
        "Beer": beer_name,
        "ML Stockouts": stockouts_ml,
        "Naive Stockouts": stockouts_naive,
        "ML Shortage (L)": shortage_ml,
        "Naive Shortage (L)": shortage_naive,
        "ML Tanks Used": tanks_ml,
        "Naive Tanks Used": tanks_naive
    }


# ==============================
# MAIN
# ==============================

def main():

    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["week"] = pd.to_datetime(df["week"])

    results = []

    for beer in df["beer"].unique():
        print(f"Simulating {beer}...")
        beer_df = df[df["beer"] == beer].copy()
        result = simulate_beer(beer_df, beer)
        results.append(result)

    results_df = pd.DataFrame(results)

    print("\n==========================================")
    print("CUMULATIVE FORECAST OPERATIONAL BACKTEST")
    print("==========================================\n")
    print(results_df)


if __name__ == "__main__":
    main()