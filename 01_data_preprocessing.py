import pandas as pd
import numpy as np
import requests
import os

# ==============================
# CONFIG
# ==============================

CORE_BEERS = [
    "Hoop US Lager",
    "Hoop Bleke Nelis",
    "Hoop BulleBier",
    "Hoop Kaper North East IPA",
    "Hoop Anker"
]

RAW_DATA_PATH = "data/raw/sales_data.csv"
PROCESSED_DATA_PATH = "data/processed/weekly_beer_data.csv"

WEATHER_LAT = 52.3676  # Amsterdam
WEATHER_LON = 4.9041


# ==============================
# LOAD DATA
# ==============================

def load_raw_data():
    print("Loading raw sales data...")

    df = pd.read_csv(
        RAW_DATA_PATH,
        sep=";",
        encoding="latin1"
    )

    # Parse Dutch date format
    df["Factuurdatum"] = pd.to_datetime(
        df["Factuurdatum"],
        dayfirst=True
    )

    return df


# ==============================
# FILTER CORE BEERS
# ==============================

def filter_core_beers(df):
    print("Filtering core beers...")
    return df[df["Grondstof"].isin(CORE_BEERS)].copy()


# ==============================
# WEEKLY AGGREGATION
# ==============================

def aggregate_weekly(df):
    print("Aggregating to weekly level...")

    df["week"] = df["Factuurdatum"].dt.to_period("W").apply(lambda r: r.start_time)

    weekly = (
        df.groupby(["week", "Grondstof"])["Liter"]
        .sum()
        .reset_index()
    )

    weekly.rename(columns={
        "Grondstof": "beer",
        "Liter": "liters"
    }, inplace=True)

    return weekly


# ==============================
# FILL MISSING WEEKS
# ==============================

def create_full_timeline(weekly):
    print("Creating continuous weekly timeline...")

    beers = weekly["beer"].unique()

    full_range = pd.date_range(
        weekly["week"].min(),
        weekly["week"].max(),
        freq="W-MON"
    )

    full_index = pd.MultiIndex.from_product(
        [full_range, beers],
        names=["week", "beer"]
    )

    full_df = (
        weekly.set_index(["week", "beer"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    return full_df


# ==============================
# WEATHER DATA
# ==============================

def fetch_weather_data(start_date, end_date):
    print("Fetching historical weather data...")

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={WEATHER_LAT}"
        f"&longitude={WEATHER_LON}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        "&daily=temperature_2m_mean,precipitation_sum"
        "&timezone=Europe%2FBerlin"
    )

    response = requests.get(url)

    if response.status_code != 200:
        print("⚠ Weather API failed. Continuing without weather.")
        return None

    data = response.json()

    weather_df = pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "temp_mean": data["daily"]["temperature_2m_mean"],
        "rain_mm": data["daily"]["precipitation_sum"]
    })

    weather_df["week"] = weather_df["date"].dt.to_period("W").apply(lambda r: r.start_time)

    weekly_weather = (
        weather_df.groupby("week")
        .agg({
            "temp_mean": "mean",
            "rain_mm": "sum"
        })
        .reset_index()
    )

    return weekly_weather


# ==============================
# FEATURE ENGINEERING
# ==============================

def add_time_features(df):
    print("Engineering time features...")

    df["week_of_year"] = df["week"].dt.isocalendar().week.astype(int)
    df["month"] = df["week"].dt.month
    df["year"] = df["week"].dt.year

    df = df.sort_values(["beer", "week"])

    df["time_index"] = (
        (df["week"] - df["week"].min()).dt.days // 7
    )

    # Weather flags (if weather exists)
    if "temp_mean" in df.columns:
        df["hot_week"] = (df["temp_mean"] > 20).astype(int)
        df["rainy_week"] = (df["rain_mm"] > 10).astype(int)
    else:
        df["hot_week"] = 0
        df["rainy_week"] = 0

    return df


# ==============================
# SAVE OUTPUT
# ==============================

def save_processed(df):
    print("Saving processed dataset...")

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

    df.to_csv(PROCESSED_DATA_PATH, index=False)


# ==============================
# MAIN PIPELINE
# ==============================

def main():
    df = load_raw_data()
    df = filter_core_beers(df)
    weekly = aggregate_weekly(df)
    weekly = create_full_timeline(weekly)

    weather = fetch_weather_data(
        weekly["week"].min().strftime("%Y-%m-%d"),
        weekly["week"].max().strftime("%Y-%m-%d")
    )

    if weather is not None:
        merged = weekly.merge(weather, on="week", how="left")
    else:
        merged = weekly

    merged = add_time_features(merged)

    save_processed(merged)

    print("STEP 1 COMPLETE ✅")


if __name__ == "__main__":
    main()