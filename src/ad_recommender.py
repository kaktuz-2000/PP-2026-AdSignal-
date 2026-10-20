from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "processed" / "ads.sqlite3"
CATEGORIES = ["technology", "food", "travel", "sports", "education"]
SEGMENTS = ["student", "professional", "creator", "family"]


def create_sources(seed: int = 42, days: int = 90) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create reproducible event-log and ad-catalog sources for the demo."""
    rng = np.random.default_rng(seed)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    users = pd.DataFrame(
        {
            "user_id": [f"user_{i:03d}" for i in range(80)],
            "segment": [SEGMENTS[i % len(SEGMENTS)] for i in range(80)],
        }
    )
    ads = pd.DataFrame(
        [
            {
                "ad_id": f"ad_{category}_{variant}",
                "category": category,
                "title": title,
                "advertiser": advertiser,
                "quality": float(rng.uniform(0.75, 1.25)),
            }
            for category, entries in {
                "technology": [("Build something", "Northstar"), ("Tools for ideas", "Orbit"), ("Tech, made useful", "Circuit")],
                "food": [("A better lunch break", "Spoon"), ("Discover your next meal", "Table"), ("Fresh from the kitchen", "Market")],
                "travel": [("Take the scenic route", "Waypoint"), ("Your next weekend", "Roam"), ("Find somewhere new", "Atlas")],
                "sports": [("Move at your pace", "Stride"), ("Gear up for today", "Motion"), ("Make room to play", "Field")],
                "education": [("Learn one more thing", "Openbook"), ("Make skills practical", "Workshop"), ("A course for your curiosity", "Bright")],
            }.items()
            for variant, (title, advertiser) in enumerate(entries, start=1)
        ]
    )

    user_segments = users.set_index("user_id")["segment"].to_dict()
    segment_users = {
        segment: users.loc[users.segment == segment, "user_id"].to_numpy()
        for segment in SEGMENTS
    }
    category_index = {category: i for i, category in enumerate(CATEGORIES)}
    affinities = {
        user_id: rng.dirichlet(np.full(len(CATEGORIES), 1.4))
        for user_id in users.user_id
    }
    ad_quality = ads.set_index("ad_id")["quality"].to_dict()
    ad_options = {
        category: ads.loc[ads.category == category, "ad_id"].to_numpy()
        for category in CATEGORIES
    }

    rows: list[dict[str, object]] = []
    dates = pd.date_range(start="2026-07-01", periods=days)
    event_number = 0
    for date in dates:
        day_factor = 0.85 + 0.3 * (date.dayofweek in (4, 5, 6))
        for segment in SEGMENTS:
            for category in CATEGORIES:
                for _ in range(24):
                    user_id = str(rng.choice(segment_users[segment]))
                    ad_id = str(rng.choice(ad_options[category]))
                    affinity = affinities[user_id][category_index[category]]
                    click_probability = np.clip(
                        (0.025 + affinity * 0.15) * day_factor * ad_quality[ad_id],
                        0.01,
                        0.32,
                    )
                    clicked = int(rng.random() < click_probability)
                    liked = int(clicked and rng.random() < 0.27)
                    reposted = int(clicked and rng.random() < 0.10)
                    rows.append(
                        {
                            "event_id": f"event_{event_number:07d}",
                            "timestamp": date + pd.Timedelta(minutes=int(rng.integers(0, 1440))),
                            "user_id": user_id,
                            "segment": user_segments[user_id],
                            "ad_id": ad_id,
                            "category": category,
                            "viewed": 1,
                            "clicked": clicked,
                            "liked": liked,
                            "reposted": reposted,
                        }
                    )
                    event_number += 1

    events = pd.DataFrame(rows)
    events.to_csv(RAW_DIR / "interaction_events.csv", index=False)
    ads.drop(columns="quality").to_csv(RAW_DIR / "ad_catalog.csv", index=False)
    return events, ads.drop(columns="quality")


def load_sources(events: pd.DataFrame, ads: pd.DataFrame) -> None:
    """Idempotently load event and catalog sources into SQLite."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as connection, connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY, timestamp TEXT, user_id TEXT,
                segment TEXT, ad_id TEXT, category TEXT, viewed INTEGER,
                clicked INTEGER, liked INTEGER, reposted INTEGER
            )"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS ads (
                ad_id TEXT PRIMARY KEY, category TEXT, title TEXT, advertiser TEXT
            )"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS load_log (
                source TEXT, loaded_at TEXT, rows_received INTEGER, rows_added INTEGER
            )"""
        )
        before = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        events.to_sql("events_staging", connection, if_exists="replace", index=False)
        connection.execute(
            """INSERT OR IGNORE INTO events
               SELECT event_id, timestamp, user_id, segment, ad_id, category,
                      viewed, clicked, liked, reposted FROM events_staging"""
        )
        after = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        connection.execute("DROP TABLE events_staging")
        ads.to_sql("ads_staging", connection, if_exists="replace", index=False)
        connection.execute("INSERT OR REPLACE INTO ads SELECT * FROM ads_staging")
        connection.execute("DROP TABLE ads_staging")
        loaded_at = pd.Timestamp.now(tz="UTC").isoformat()
        connection.executemany(
            "INSERT INTO load_log VALUES (?, ?, ?, ?)",
            [
                ("interaction_events", loaded_at, len(events), after - before),
                ("ad_catalog", loaded_at, len(ads), len(ads)),
            ],
        )


def read_sources(seed: int = 42, days: int = 90) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not DB_PATH.exists():
        events, ads = create_sources(seed=seed, days=days)
        load_sources(events, ads)
    with closing(sqlite3.connect(DB_PATH)) as connection, connection:
        events = pd.read_sql_query("SELECT * FROM events", connection, parse_dates=["timestamp"])
        ads = pd.read_sql_query("SELECT * FROM ads", connection)
        loads = pd.read_sql_query("SELECT * FROM load_log ORDER BY loaded_at DESC", connection)
    return events, ads, loads


def _daily_data(events: pd.DataFrame) -> pd.DataFrame:
    data = events.copy()
    data["date"] = pd.to_datetime(data["timestamp"]).dt.floor("D")
    daily = (
        data.groupby(["date", "segment", "category"], as_index=False)
        .agg(impressions=("viewed", "sum"), clicks=("clicked", "sum"))
        .sort_values(["segment", "category", "date"])
    )
    daily["ctr"] = daily["clicks"] / daily["impressions"].clip(lower=1)
    grouped = daily.groupby(["segment", "category"], sort=False)
    daily["ctr_lag_1"] = grouped["ctr"].shift(1)
    daily["ctr_mean_7"] = grouped["ctr"].transform(
        lambda series: series.shift(1).rolling(7, min_periods=2).mean()
    )
    daily["impressions_lag_1"] = grouped["impressions"].shift(1)
    daily["target"] = grouped["ctr"].shift(-1)
    daily["target_date"] = daily["date"] + pd.Timedelta(days=1)
    daily["target_weekday"] = daily["target_date"].dt.dayofweek
    daily["target_weekend"] = daily["target_weekday"].isin([5, 6]).astype(int)
    return daily.dropna(subset=["target", "ctr_lag_1", "ctr_mean_7"])


def train_models(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Compare a naive baseline, linear regression, and random forest over time."""
    daily = _daily_data(events)
    columns = [
        "segment", "category", "ctr_lag_1", "ctr_mean_7", "impressions_lag_1",
        "target_weekday", "target_weekend",
    ]
    encoded = pd.get_dummies(daily[columns], columns=["segment", "category"], dtype=float)
    cutoff = daily["target_date"].sort_values().iloc[int(len(daily) * 0.8)]
    train_mask = daily["target_date"] < cutoff
    test_mask = ~train_mask
    if train_mask.sum() < 20 or test_mask.sum() < 5:
        raise ValueError("Недостаточно истории для временной проверки модели.")

    train_x, test_x = encoded.loc[train_mask], encoded.loc[test_mask]
    train_y, test_y = daily.loc[train_mask, "target"], daily.loc[test_mask, "target"]
    candidates = {
        "Наивная (CTR вчера)": None,
        "Линейная регрессия": LinearRegression(),
        "Случайный лес": RandomForestRegressor(
            n_estimators=120, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
    }
    metrics: list[dict[str, float | str]] = []
    for name, model in candidates.items():
        if model is None:
            prediction = daily.loc[test_mask, "ctr_lag_1"].to_numpy()
        else:
            model.fit(train_x, train_y)
            prediction = model.predict(test_x)
        prediction = np.clip(prediction, 0, 1)
        metrics.append(
            {
                "Модель": name,
                "MAE": float(mean_absolute_error(test_y, prediction)),
                "RMSE": float(np.sqrt(mean_squared_error(test_y, prediction))),
                "Клики на 1000 показов": float(np.mean(prediction) * 1000),
            }
        )

    best_name = min(metrics, key=lambda row: row["MAE"])["Модель"]
    best_model = candidates[best_name]
    latest = (
        daily.sort_values("date")
        .groupby(["segment", "category"], as_index=False)
        .tail(1)
        .copy()
    )
    forecast_rows = latest[columns].copy()
    forecast_rows["target_weekday"] = (latest["date"] + pd.Timedelta(days=1)).dt.dayofweek.to_numpy()
    forecast_rows["target_weekend"] = forecast_rows["target_weekday"].isin([5, 6]).astype(int)
    forecast_x = pd.get_dummies(forecast_rows, columns=["segment", "category"], dtype=float)
    forecast_x = forecast_x.reindex(columns=encoded.columns, fill_value=0)
    if best_model is None:
        forecast = latest["ctr"].to_numpy()
    else:
        best_model.fit(encoded, daily["target"])
        forecast = best_model.predict(forecast_x)
    forecasts = latest[["segment", "category", "date"]].copy()
    forecasts["forecast_date"] = forecasts["date"] + pd.Timedelta(days=1)
    forecasts["forecast_ctr"] = np.clip(forecast, 0, 1)
    forecasts["expected_clicks_per_1000"] = forecasts["forecast_ctr"] * 1000
    return pd.DataFrame(metrics).sort_values("MAE"), forecasts, str(best_name)


def recommend_ads(
    user_id: str, events: pd.DataFrame, ads: pd.DataFrame, forecasts: pd.DataFrame
) -> pd.DataFrame:
    history = events.loc[events["user_id"] == user_id]
    if history.empty:
        return ads.assign(score=0.0, forecast_ctr=0.0)
    segment = str(history["segment"].mode().iloc[0])
    preferences = (
        history.groupby("category")
        .agg(views=("viewed", "sum"), clicks=("clicked", "sum"), likes=("liked", "sum"), reposts=("reposted", "sum"))
    )
    preferences["interest"] = preferences["views"] + 2 * preferences["clicks"] + 3 * preferences["likes"] + 4 * preferences["reposts"]
    preferences["interest"] /= max(float(preferences["interest"].max()), 1.0)
    segment_forecasts = forecasts.loc[forecasts["segment"] == segment, ["category", "forecast_ctr"]]
    ranked = ads.merge(segment_forecasts, on="category", how="left").merge(
        preferences[["interest"]], left_on="category", right_index=True, how="left"
    )
    ranked["interest"] = ranked["interest"].fillna(0)
    ranked["forecast_ctr"] = ranked["forecast_ctr"].fillna(0)
    ranked["score"] = ranked["forecast_ctr"] * (0.35 + 0.65 * ranked["interest"])
    ranked["forecast_clicks_per_1000"] = ranked["forecast_ctr"] * 1000
    return ranked.sort_values("score", ascending=False).reset_index(drop=True)