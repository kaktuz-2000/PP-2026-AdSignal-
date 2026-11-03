from __future__ import annotations

import hashlib
import gzip
import io
import json
import pickle
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_selection import mutual_info_regression


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "processed" / "ads.sqlite3"
CATEGORIES = ["technology", "food", "travel", "sports", "education"]
SEGMENTS = ["student", "professional", "creator", "family"]
EVENT_COLUMNS = [
    "event_id", "timestamp", "user_id", "segment", "ad_id", "category",
    "viewed", "clicked", "liked", "reposted",
]
AD_COLUMNS = ["ad_id", "category", "title", "advertiser"]
FEATURE_COLUMNS = [
    "segment", "category", "ctr_lag_1", "ctr_mean_7", "impressions_lag_1",
    "target_weekday", "target_weekend",
]
PIPELINE_VERSION = 3
NUMERIC_FEATURES = [
    "ctr_lag_1", "ctr_mean_7", "impressions_lag_1",
    "target_weekday", "target_weekend",
]
RF_CONFIGURATIONS = [
    {"n_estimators": 80, "max_depth": 6, "min_samples_leaf": 3},
    {"n_estimators": 120, "max_depth": 8, "min_samples_leaf": 3},
    {"n_estimators": 160, "max_depth": 10, "min_samples_leaf": 2},
    {"n_estimators": 120, "max_depth": None, "min_samples_leaf": 5},
]


def _ensure_schema(connection: sqlite3.Connection) -> None:
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
    connection.execute(
        """CREATE TABLE IF NOT EXISTS quality_log (
            run_id TEXT, checked_at TEXT, severity TEXT, check_name TEXT,
            passed INTEGER, details TEXT
        )"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS experiments (
            experiment_id TEXT PRIMARY KEY, created_at TEXT, model_name TEXT,
            parameters TEXT, seed INTEGER, data_hash TEXT, train_rows INTEGER,
            validation_mae REAL, validation_rmse REAL, selected INTEGER
        )"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS model_registry (
            version_id TEXT PRIMARY KEY, created_at TEXT, model_name TEXT,
            parameters TEXT, features TEXT, seed INTEGER, data_hash TEXT,
            validation_mae REAL, holdout_mae REAL, artifact BLOB,
            status TEXT
        )"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS active_model (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            version_id TEXT NOT NULL
        )"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS monitoring_log (
            run_id TEXT PRIMARY KEY, created_at TEXT, version_id TEXT,
            maximum_psi REAL, recent_mae REAL, baseline_mae REAL,
            drift_detected INTEGER, retrain_recommended INTEGER, details TEXT
        )"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS evaluation_runs (
            run_id TEXT PRIMARY KEY, created_at TEXT, version_id TEXT,
            data_hash TEXT, metrics TEXT
        )"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS dataset_snapshots (
            data_hash TEXT PRIMARY KEY, created_at TEXT, events_gzip BLOB
        )"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS experiment_replays (
            replay_id TEXT PRIMARY KEY, original_experiment_id TEXT,
            replayed_at TEXT, data_hash TEXT, validation_mae REAL,
            validation_rmse REAL, holdout_mae REAL, holdout_rmse REAL
        )"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS feature_store (
            data_hash TEXT NOT NULL, date TEXT NOT NULL, segment TEXT NOT NULL,
            category TEXT NOT NULL, impressions REAL, clicks REAL, ctr REAL,
            ctr_lag_1 REAL, ctr_mean_7 REAL, impressions_lag_1 REAL,
            target REAL, target_date TEXT, target_weekday INTEGER,
            target_weekend INTEGER,
            PRIMARY KEY (data_hash, date, segment, category)
        )"""
    )


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    _ensure_schema(connection)
    return connection


def validate_sources(events: pd.DataFrame, ads: pd.DataFrame) -> pd.DataFrame:
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, details: str, severity: str = "error") -> None:
        checks.append(
            {"Проверка": name, "Результат": "OK" if passed else severity.upper(), "Детали": details}
        )

    missing_events = sorted(set(EVENT_COLUMNS) - set(events.columns))
    missing_ads = sorted(set(AD_COLUMNS) - set(ads.columns))
    add("Колонки журнала событий", not missing_events, f"Отсутствуют: {missing_events}" if missing_events else "Все обязательные колонки на месте.")
    add("Колонки каталога объявлений", not missing_ads, f"Отсутствуют: {missing_ads}" if missing_ads else "Все обязательные колонки на месте.")
    if missing_events or missing_ads:
        return pd.DataFrame(checks)

    required_event_values = ["event_id", "timestamp", "user_id", "segment", "ad_id", "category"]
    missing_values = int(events[required_event_values].isna().any(axis=1).sum())
    add("Обязательные значения событий", missing_values == 0, f"Строк с пустыми обязательными полями: {missing_values}.")
    missing_ad_values = int(ads[AD_COLUMNS].isna().any(axis=1).sum())
    add("Обязательные значения каталога", missing_ad_values == 0, f"Строк с пустыми обязательными полями: {missing_ad_values}.")

    duplicate_events = int(events["event_id"].duplicated(keep=False).sum())
    add("Уникальность event_id", duplicate_events == 0, f"Строк с повторяющимся event_id: {duplicate_events}.")
    duplicate_ads = int(ads["ad_id"].duplicated(keep=False).sum())
    add("Уникальность ad_id", duplicate_ads == 0, f"Строк с повторяющимся ad_id: {duplicate_ads}.")

    timestamps = pd.to_datetime(events["timestamp"], errors="coerce", utc=True)
    invalid_timestamps = int(timestamps.isna().sum())
    add("Корректность дат", invalid_timestamps == 0, f"Некорректных дат: {invalid_timestamps}.")

    binary_columns = ["viewed", "clicked", "liked", "reposted"]
    binary_values_ok = True
    for column in binary_columns:
        values = pd.to_numeric(events[column], errors="coerce")
        binary_values_ok = binary_values_ok and values.notna().all() and values.isin([0, 1]).all()
    add("Бинарные признаки событий", bool(binary_values_ok), "Поля viewed/clicked/liked/reposted должны содержать только 0 или 1.")

    counts = events[binary_columns].apply(pd.to_numeric, errors="coerce")
    impressions_ok = bool((counts["clicked"] <= counts["viewed"]).all())
    impressions_ok = impressions_ok and bool((counts["liked"] <= counts["clicked"]).all())
    impressions_ok = impressions_ok and bool((counts["reposted"] <= counts["clicked"]).all())
    add("Логическая согласованность событий", impressions_ok, "Клик не может быть без показа; лайк и репост — без клика.")

    ad_lookup = ads.set_index("ad_id")["category"].to_dict() if duplicate_ads == 0 else {}
    unknown_ads = int((~events["ad_id"].isin(ad_lookup)).sum()) if ad_lookup else len(events)
    category_mismatches = int(
        sum(
            ad_lookup.get(row.ad_id) != row.category
            for row in events[["ad_id", "category"]].itertuples(index=False)
            if row.ad_id in ad_lookup
        )
    )
    add("Связь событий с каталогом", unknown_ads == 0 and category_mismatches == 0, f"Неизвестных объявлений: {unknown_ads}; несовпадений категорий: {category_mismatches}.")

    date_count = int(timestamps.dt.date.nunique())
    add("История для обучения", date_count >= 11, f"Уникальных календарных дней: {date_count}; для временной проверки требуется не менее 11.")
    return pd.DataFrame(checks)


def _record_quality(connection: sqlite3.Connection, report: pd.DataFrame) -> None:
    run_id = str(uuid.uuid4())
    checked_at = datetime.now(timezone.utc).isoformat()
    connection.executemany(
        "INSERT INTO quality_log VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                run_id,
                checked_at,
                str(row["Результат"]),
                str(row["Проверка"]),
                int(row["Результат"] == "OK"),
                str(row["Детали"]),
            )
            for _, row in report.iterrows()
        ],
    )


def create_sources(
    seed: int = 42, days: int = 90, drift: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create reproducible event-log and ad-catalog sources for the demo."""
    if days < 10:
        raise ValueError("Для временной проверки требуется не менее 10 дней данных.")
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
    dates = pd.date_range(end="2026-11-01", periods=days)
    event_number = 0
    for day_index, date in enumerate(dates):
        day_factor = 0.85 + 0.3 * (date.dayofweek in (4, 5, 6))
        drift_factor = 0.35 if drift and day_index >= days // 2 else 1.0
        for segment in SEGMENTS:
            for category in CATEGORIES:
                for _ in range(24):
                    user_id = str(rng.choice(segment_users[segment]))
                    ad_id = str(rng.choice(ad_options[category]))
                    affinity = affinities[user_id][category_index[category]]
                    click_probability = np.clip(
                        (0.025 + affinity * 0.15) * day_factor * ad_quality[ad_id] * drift_factor,
                        0.003,
                        0.32,
                    )
                    clicked = int(rng.random() < click_probability)
                    liked = int(clicked and rng.random() < 0.27)
                    reposted = int(clicked and rng.random() < 0.10)
                    rows.append(
                        {
                            "event_id": f"event_{seed}_{event_number:07d}",
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

    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    ad_catalog = ads.drop(columns="quality")[AD_COLUMNS]
    events.to_csv(RAW_DIR / "interaction_events.csv", index=False)
    ad_catalog.to_csv(RAW_DIR / "ad_catalog.csv", index=False)
    return events, ad_catalog


def load_sources(
    events: pd.DataFrame, ads: pd.DataFrame, replace: bool = False
) -> None:
    report = validate_sources(events, ads)
    with closing(_connect()) as connection, connection:
        _record_quality(connection, report)
        failed = report.loc[report["Результат"] == "ERROR"]
        if not failed.empty:
            checks = "; ".join(failed["Проверка"].astype(str))
            connection.commit()
            raise ValueError(f"Данные не прошли проверку качества: {checks}")
        before = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        if replace:
            connection.execute("DELETE FROM events")
            connection.execute("DELETE FROM ads")
            before = 0
        event_rows = events.loc[:, EVENT_COLUMNS].copy()
        ad_rows = ads.loc[:, AD_COLUMNS].copy()
        for column in ("event_id", "user_id", "segment", "ad_id", "category"):
            event_rows[column] = event_rows[column].astype(str)
        for column in AD_COLUMNS:
            ad_rows[column] = ad_rows[column].astype(str)
        event_rows.to_sql(
            "events_staging", connection, if_exists="replace", index=False
        )
        connection.execute(
            """INSERT OR IGNORE INTO events
               SELECT event_id, timestamp, user_id, segment, ad_id, category,
                      viewed, clicked, liked, reposted FROM events_staging"""
        )
        after = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        connection.execute("DROP TABLE events_staging")
        ad_rows.to_sql(
            "ads_staging", connection, if_exists="replace", index=False
        )
        connection.execute(
            """INSERT INTO ads SELECT * FROM ads_staging WHERE 1
               ON CONFLICT(ad_id) DO UPDATE SET
               category=excluded.category, title=excluded.title,
               advertiser=excluded.advertiser"""
        )
        connection.execute("DROP TABLE ads_staging")
        loaded_at = datetime.now(timezone.utc).isoformat()
        connection.executemany(
            "INSERT INTO load_log VALUES (?, ?, ?, ?)",
            [
                ("interaction_events", loaded_at, len(events), after - before),
                ("ad_catalog", loaded_at, len(ads), len(ads)),
            ],
        )


def reset_demo_data(seed: int = 42, days: int = 90, drift: bool = False) -> None:
    """Replace demo input data while retaining model, experiment, and monitoring history."""
    events, ads = create_sources(seed=seed, days=days, drift=drift)
    load_sources(events, ads, replace=True)


def read_sources(
    seed: int = 42, days: int = 90
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not DB_PATH.exists():
        events, ads = create_sources(seed=seed, days=days)
        load_sources(events, ads)
    with closing(_connect()) as connection, connection:
        event_count = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        if event_count == 0:
            events, ads = create_sources(seed=seed, days=days)
            load_sources(events, ads)
        events = pd.read_sql_query(
            "SELECT * FROM events ORDER BY timestamp", connection, parse_dates=["timestamp"]
        )
        ads = pd.read_sql_query("SELECT * FROM ads ORDER BY ad_id", connection)
        loads = pd.read_sql_query(
            "SELECT * FROM load_log ORDER BY loaded_at DESC", connection
        )
    return events, ads, loads


def _daily_data(
    events: pd.DataFrame, include_unlabeled: bool = False
) -> pd.DataFrame:
    if events.empty:
        raise ValueError("Нет событий для построения признаков.")
    data = events.copy()
    data["date"] = pd.to_datetime(data["timestamp"], errors="coerce").dt.floor("D")
    daily = (
        data.groupby(["date", "segment", "category"], as_index=False)
        .agg(impressions=("viewed", "sum"), clicks=("clicked", "sum"))
        .sort_values(["segment", "category", "date"])
    )
    all_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    pairs = daily[["segment", "category"]].drop_duplicates()
    full_grid = pairs.merge(pd.DataFrame({"date": all_dates}), how="cross")
    daily = full_grid.merge(daily, on=["date", "segment", "category"], how="left")
    daily[["impressions", "clicks"]] = daily[["impressions", "clicks"]].fillna(0)
    daily = daily.sort_values(["segment", "category", "date"])
    daily["ctr"] = daily["clicks"].div(daily["impressions"].where(daily["impressions"] > 0))
    grouped = daily.groupby(["segment", "category"], sort=False)
    prior_ctr = grouped["ctr"].transform(lambda series: series.ffill().shift(1))
    daily["ctr_lag_1"] = prior_ctr
    daily["ctr_mean_7"] = grouped["ctr"].transform(
        lambda series: series.ffill().shift(1).rolling(7, min_periods=2).mean()
    )
    daily["impressions_lag_1"] = grouped["impressions"].shift(1)
    daily["target"] = grouped["ctr"].shift(-1)
    daily["target_date"] = daily["date"] + pd.Timedelta(days=1)
    daily["target_weekday"] = daily["target_date"].dt.dayofweek
    daily["target_weekend"] = daily["target_weekday"].isin([5, 6]).astype(int)
    required = ["ctr_lag_1", "ctr_mean_7"]
    if not include_unlabeled:
        required.append("target")
    return daily.dropna(subset=required)


def _split_dates(daily: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    target_dates = pd.DatetimeIndex(sorted(daily["target_date"].unique()))
    if len(target_dates) < 10:
        raise ValueError("Для временной проверки требуется не менее 10 целевых дат.")
    validation_start = target_dates[max(1, int(len(target_dates) * 0.60))]
    holdout_start = target_dates[max(2, int(len(target_dates) * 0.80))]
    if validation_start >= holdout_start:
        raise ValueError("Недостаточно данных для отдельных validation и holdout периодов.")
    return pd.Timestamp(validation_start), pd.Timestamp(holdout_start)


def _encoded_features(daily: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(daily[FEATURE_COLUMNS], columns=["segment", "category"], dtype=float)


def _select_feature_columns(
    train_x: pd.DataFrame, train_y: pd.Series, seed: int
) -> list[str]:
    selector = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=1,
    )
    selector.fit(train_x, train_y)
    ranked = np.argsort(selector.feature_importances_)[::-1]
    cumulative = np.cumsum(selector.feature_importances_[ranked])
    selected_count = max(3, int(np.searchsorted(cumulative, 0.90, side="left")) + 1)
    selected = {train_x.columns[index] for index in ranked[:selected_count]}
    return [column for column in train_x.columns if column in selected]


def _new_model(model_name: str, parameters: dict[str, Any], seed: int):
    parameters = {
        key: value
        for key, value in parameters.items()
        if key not in {"feature_set", "features"}
    }
    if model_name == "Линейная регрессия":
        return LinearRegression()
    if model_name == "Случайный лес":
        return RandomForestRegressor(
            **parameters, random_state=seed, n_jobs=1
        )
    if model_name == "Наивная (CTR вчера)":
        return None
    raise ValueError(f"Неизвестная модель: {model_name}")


def _predict(
    model_name: str, model: Any, features: pd.DataFrame
) -> np.ndarray:
    if model_name == "Наивная (CTR вчера)":
        prediction = features["ctr_lag_1"].to_numpy(dtype=float)
    else:
        prediction = model.predict(features)
    return np.clip(np.asarray(prediction, dtype=float), 0, 1)


def _dataset_hash(events: pd.DataFrame) -> str:
    ordered = events.loc[:, EVENT_COLUMNS].sort_values("event_id")
    return hashlib.sha256(ordered.to_csv(index=False).encode("utf-8")).hexdigest()


def _persist_features(
    connection: sqlite3.Connection, daily: pd.DataFrame, data_hash: str
) -> None:
    columns = [
        "date", "segment", "category", "impressions", "clicks", "ctr",
        "ctr_lag_1", "ctr_mean_7", "impressions_lag_1", "target",
        "target_date", "target_weekday", "target_weekend",
    ]
    stored = daily.loc[:, columns].copy()
    stored.insert(0, "data_hash", data_hash)
    stored["date"] = pd.to_datetime(stored["date"]).dt.strftime("%Y-%m-%d")
    stored["target_date"] = pd.to_datetime(stored["target_date"]).dt.strftime("%Y-%m-%d")
    stored.to_sql("feature_store_staging", connection, if_exists="replace", index=False)
    connection.execute(
        """INSERT OR REPLACE INTO feature_store
           SELECT data_hash, date, segment, category, impressions, clicks, ctr,
                  ctr_lag_1, ctr_mean_7, impressions_lag_1, target, target_date,
                  target_weekday, target_weekend
           FROM feature_store_staging"""
    )
    connection.execute("DROP TABLE feature_store_staging")


def data_fingerprint(events: pd.DataFrame) -> str:
    return _dataset_hash(events)


def _active_model_row(connection: sqlite3.Connection) -> sqlite3.Row | None:
    connection.row_factory = sqlite3.Row
    return connection.execute(
        """SELECT registry.* FROM active_model
           JOIN model_registry AS registry USING(version_id)
           WHERE active_model.singleton = 1"""
    ).fetchone()


def _load_active_bundle(connection: sqlite3.Connection) -> dict[str, Any] | None:
    row = _active_model_row(connection)
    if row is None:
        return None
    return pickle.loads(row["artifact"])


def _add_experiment(
    connection: sqlite3.Connection,
    model_name: str,
    parameters: dict[str, Any],
    seed: int,
    data_hash: str,
    train_rows: int,
    validation_mae: float,
    validation_rmse: float,
    selected: bool,
) -> None:
    connection.execute(
        """INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            str(uuid.uuid4()),
            datetime.now(timezone.utc).isoformat(),
            model_name,
            json.dumps(parameters, sort_keys=True),
            seed,
            data_hash,
            train_rows,
            validation_mae,
            validation_rmse,
            int(selected),
        ),
    )


def _register_model(
    connection: sqlite3.Connection,
    bundle: dict[str, Any],
    data_hash: str,
    validation_mae: float,
    holdout_mae: float,
) -> str:
    prior = _active_model_row(connection)
    if prior is not None:
        connection.execute(
            "UPDATE model_registry SET status='archived' WHERE version_id=?",
            (prior["version_id"],),
        )
    version_id = f"model-{datetime.now(timezone.utc):%Y%m%d%H%M%S}-{uuid.uuid4().hex[:8]}"
    connection.execute(
        """INSERT INTO model_registry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            version_id,
            datetime.now(timezone.utc).isoformat(),
            bundle["model_name"],
            json.dumps(bundle["parameters"], sort_keys=True),
            json.dumps(bundle["columns"]),
            bundle["seed"],
            data_hash,
            validation_mae,
            holdout_mae,
            sqlite3.Binary(pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)),
            "active",
        ),
    )
    connection.execute(
        """INSERT INTO active_model(singleton, version_id) VALUES (1, ?)
           ON CONFLICT(singleton) DO UPDATE SET version_id=excluded.version_id""",
        (version_id,),
    )
    return version_id


def train_models(
    events: pd.DataFrame,
    seed: int = 42,
    search_budget: int = 4,
    promotion_threshold: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if not 1 <= search_budget <= len(RF_CONFIGURATIONS):
        raise ValueError(f"Бюджет поиска должен быть от 1 до {len(RF_CONFIGURATIONS)}.")
    if not 0 <= promotion_threshold < 1:
        raise ValueError("Порог улучшения модели должен быть в диапазоне [0, 1).")
    quality = validate_sources(
        events, pd.DataFrame(
            events[["ad_id", "category"]].drop_duplicates()
            .assign(title="event", advertiser="source")
        )
    )
    errors = quality.loc[quality["Результат"] == "ERROR"]
    if not errors.empty:
        raise ValueError("Данные для обучения не прошли контроль качества.")

    daily = _daily_data(events)
    encoded = _encoded_features(daily)
    validation_start, holdout_start = _split_dates(daily)
    train_mask = daily["target_date"] < validation_start
    validation_mask = (daily["target_date"] >= validation_start) & (
        daily["target_date"] < holdout_start
    )
    holdout_mask = daily["target_date"] >= holdout_start
    if train_mask.sum() < 20 or validation_mask.sum() < 5 or holdout_mask.sum() < 5:
        raise ValueError("Недостаточно примеров для обучения, подбора и отдельной holdout-проверки.")

    train_x = encoded.loc[train_mask]
    validation_x = encoded.loc[validation_mask]
    holdout_x = encoded.loc[holdout_mask]
    train_y = daily.loc[train_mask, "target"]
    validation_y = daily.loc[validation_mask, "target"]
    holdout_y = daily.loc[holdout_mask, "target"]
    data_hash = _dataset_hash(events)
    all_daily_features = _daily_data(events, include_unlabeled=True)
    all_columns = list(encoded.columns)
    selected_columns = _select_feature_columns(train_x, train_y, seed)

    candidates: list[dict[str, Any]] = [
        {
            "name": "Наивная (CTR вчера)",
            "parameters": {},
            "columns": all_columns,
            "feature_set": "all",
        },
        {
            "name": "Линейная регрессия",
            "parameters": {},
            "columns": all_columns,
            "feature_set": "all",
        },
        {
            "name": "Линейная регрессия",
            "parameters": {},
            "columns": selected_columns,
            "feature_set": "selected",
        },
    ]
    for params in RF_CONFIGURATIONS[:search_budget]:
        candidates.extend(
            [
                {
                    "name": "Случайный лес",
                    "parameters": params,
                    "columns": all_columns,
                    "feature_set": "all",
                },
                {
                    "name": "Случайный лес",
                    "parameters": params,
                    "columns": selected_columns,
                    "feature_set": "selected",
                },
            ]
        )
    with closing(_connect()) as connection, connection:
        ordered_events = events.loc[:, EVENT_COLUMNS].sort_values("event_id")
        _persist_features(connection, all_daily_features, data_hash)
        connection.execute(
            """INSERT OR IGNORE INTO dataset_snapshots
               VALUES (?, ?, ?)""",
            (
                data_hash,
                datetime.now(timezone.utc).isoformat(),
                sqlite3.Binary(
                    gzip.compress(ordered_events.to_csv(index=False).encode("utf-8"))
                ),
            ),
        )
        incumbent = _load_active_bundle(connection)
        if incumbent is not None:
            incumbent_config = {
                "name": incumbent["model_name"],
                "parameters": incumbent["parameters"],
                "columns": incumbent["columns"],
                "feature_set": "incumbent",
            }
            if not any(
                candidate["name"] == incumbent_config["name"]
                and candidate["parameters"] == incumbent_config["parameters"]
                and candidate["columns"] == incumbent_config["columns"]
                for candidate in candidates
            ):
                candidates.append(incumbent_config)

        results: list[dict[str, Any]] = []
        for candidate in candidates:
            model = _new_model(candidate["name"], candidate["parameters"], seed)
            candidate_train_x = train_x.loc[:, candidate["columns"]]
            candidate_validation_x = validation_x.loc[:, candidate["columns"]]
            if model is not None:
                model.fit(candidate_train_x, train_y)
            prediction = _predict(candidate["name"], model, candidate_validation_x)
            results.append(
                {
                    **candidate,
                    "model": model,
                    "validation_mae": float(mean_absolute_error(validation_y, prediction)),
                    "validation_rmse": float(np.sqrt(mean_squared_error(validation_y, prediction))),
                }
            )
        winner = min(results, key=lambda result: result["validation_mae"])
        winner_model = _new_model(winner["name"], winner["parameters"], seed)
        winner_columns = winner["columns"]
        if winner_model is not None:
            winner_model.fit(
                pd.concat([train_x, validation_x]).loc[:, winner_columns],
                pd.concat([train_y, validation_y]),
            )
        holdout_prediction = _predict(
            winner["name"], winner_model, holdout_x.loc[:, winner_columns]
        )
        holdout_mae = float(mean_absolute_error(holdout_y, holdout_prediction))

        metrics_rows: list[dict[str, float | str]] = []
        for model_name in ("Наивная (CTR вчера)", "Линейная регрессия", "Случайный лес"):
            matching = [
                result for result in results if result["name"] == model_name
            ]
            selected = min(matching, key=lambda result: result["validation_mae"])
            model = _new_model(model_name, selected["parameters"], seed)
            selected_columns_for_model = selected["columns"]
            if model is not None:
                model.fit(
                    pd.concat([train_x, validation_x]).loc[:, selected_columns_for_model],
                    pd.concat([train_y, validation_y]),
                )
            prediction = _predict(
                model_name, model, holdout_x.loc[:, selected_columns_for_model]
            )
            metrics_rows.append(
                {
                    "Модель": model_name,
                    "MAE": float(mean_absolute_error(holdout_y, prediction)),
                    "RMSE": float(np.sqrt(mean_squared_error(holdout_y, prediction))),
                    "Validation MAE": selected["validation_mae"],
                    "Выбрана по validation": model_name == winner["name"],
                    "Набор признаков": selected["feature_set"],
                    "Число признаков": len(selected_columns_for_model),
                    "Клики на 1000 показов": float(np.mean(prediction) * 1000),
                }
            )

        if incumbent is None:
            incumbent_validation_mae = float("inf")
        else:
            incumbent_features = validation_x.reindex(
                columns=incumbent["columns"], fill_value=0
            )
            incumbent_prediction = _predict(
                incumbent["model_name"], incumbent["model"], incumbent_features
            )
            incumbent_validation_mae = float(
                mean_absolute_error(validation_y, incumbent_prediction)
            )
        requires_upgrade = incumbent is not None and int(
            incumbent.get("pipeline_version", 0)
        ) < PIPELINE_VERSION
        promote = (
            incumbent is None
            or requires_upgrade
            or winner["validation_mae"]
            <= incumbent_validation_mae * (1 - promotion_threshold)
        )

        for result in results:
            _add_experiment(
                connection,
                result["name"],
                {
                    **result["parameters"],
                    "feature_set": result["feature_set"],
                    "features": result["columns"],
                },
                seed,
                data_hash,
                int(train_mask.sum()),
                result["validation_mae"],
                result["validation_rmse"],
                result is winner,
            )

        if promote:
            final_model = _new_model(winner["name"], winner["parameters"], seed)
            if final_model is not None:
                final_model.fit(encoded.loc[:, winner_columns], daily["target"])
            reference = encoded.tail(min(500, len(encoded))).to_dict(orient="list")
            bundle = {
                "model_name": winner["name"],
                "model": final_model,
                "parameters": winner["parameters"],
                "columns": winner_columns,
                "seed": seed,
                "data_hash": data_hash,
                "pipeline_version": PIPELINE_VERSION,
                "holdout_model": winner_model,
                "holdout_start": str(holdout_start),
                "reference": reference,
                "reference_dates": daily["target_date"].tail(min(500, len(daily))).astype(str).tolist(),
            }
            _register_model(
                connection, bundle, data_hash, winner["validation_mae"], holdout_mae
            )

        active_bundle = _load_active_bundle(connection)
        if active_bundle is None:
            raise RuntimeError("Обучение завершилось без зарегистрированной активной модели.")
        active_row = _active_model_row(connection)
        connection.execute(
            "INSERT INTO evaluation_runs VALUES (?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                datetime.now(timezone.utc).isoformat(),
                active_row["version_id"],
                data_hash,
                json.dumps(metrics_rows, ensure_ascii=False),
            ),
        )
        if len(results) > 0:
            for result in results:
                result["selected"] = result is winner

    forecast_daily = _daily_data(events, include_unlabeled=True)
    latest = (
        forecast_daily.sort_values("date")
        .groupby(["segment", "category"], as_index=False)
        .tail(1)
        .copy()
    )
    forecast_rows = latest[FEATURE_COLUMNS].copy()
    forecast_rows["target_weekday"] = (
        latest["date"] + pd.Timedelta(days=1)
    ).dt.dayofweek.to_numpy()
    forecast_rows["target_weekend"] = forecast_rows["target_weekday"].isin([5, 6]).astype(int)
    forecast_x = pd.get_dummies(
        forecast_rows, columns=["segment", "category"], dtype=float
    ).reindex(columns=active_bundle["columns"], fill_value=0)
    forecasts = latest[["segment", "category", "date"]].copy()
    forecasts["forecast_date"] = forecasts["date"] + pd.Timedelta(days=1)
    forecasts["forecast_ctr"] = _predict(
        active_bundle["model_name"], active_bundle["model"], forecast_x
    )
    forecasts["expected_clicks_per_1000"] = forecasts["forecast_ctr"] * 1000
    return (
        pd.DataFrame(metrics_rows).sort_values("MAE").reset_index(drop=True),
        forecasts.reset_index(drop=True),
        str(active_bundle["model_name"]),
    )


def predict_active_model(events: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    daily = _daily_data(events, include_unlabeled=True)
    with closing(_connect()) as connection:
        bundle = _load_active_bundle(connection)
    if bundle is None:
        raise RuntimeError("Сначала обучите и зарегистрируйте модель.")
    latest = (
        daily.sort_values("date")
        .groupby(["segment", "category"], as_index=False)
        .tail(1)
        .copy()
    )
    rows = latest[FEATURE_COLUMNS].copy()
    rows["target_weekday"] = (latest["date"] + pd.Timedelta(days=1)).dt.dayofweek.to_numpy()
    rows["target_weekend"] = rows["target_weekday"].isin([5, 6]).astype(int)
    features = _encoded_forecasts(rows, bundle["columns"])
    forecasts = latest[["segment", "category", "date"]].copy()
    forecasts["forecast_date"] = forecasts["date"] + pd.Timedelta(days=1)
    forecasts["forecast_ctr"] = _predict(bundle["model_name"], bundle["model"], features)
    forecasts["expected_clicks_per_1000"] = forecasts["forecast_ctr"] * 1000
    return forecasts.reset_index(drop=True), str(bundle["model_name"])


def get_latest_model_metrics() -> pd.DataFrame:
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT metrics FROM evaluation_runs ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return pd.DataFrame()
    return pd.DataFrame(json.loads(row[0]))


def active_model_info() -> dict[str, Any] | None:
    with closing(_connect()) as connection:
        row = _active_model_row(connection)
    if row is None:
        return None
    return {
        "version_id": str(row["version_id"]),
        "model_name": str(row["model_name"]),
        "data_hash": str(row["data_hash"]),
        "validation_mae": float(row["validation_mae"]),
        "holdout_mae": float(row["holdout_mae"]),
        "created_at": str(row["created_at"]),
        "pipeline_version": int(
            pickle.loads(row["artifact"]).get("pipeline_version", 0)
        ),
    }


def _encoded_forecasts(
    frame: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    return pd.get_dummies(
        frame[FEATURE_COLUMNS], columns=["segment", "category"], dtype=float
    ).reindex(columns=columns, fill_value=0)


def explain_forecast(
    events: pd.DataFrame,
    segment: str,
    category: str,
    forecast_date: str | pd.Timestamp | None = None,
) -> tuple[float, pd.DataFrame]:
    """Explain one forecast by showing one-feature-at-a-time reference changes."""
    daily = _daily_data(events, include_unlabeled=True)
    selected = daily.loc[
        (daily["segment"] == segment) & (daily["category"] == category)
    ].sort_values("date")
    if forecast_date is not None:
        requested_date = pd.Timestamp(forecast_date).normalize()
        selected = selected.loc[selected["target_date"] == requested_date]
    else:
        selected = selected.tail(1)
    if selected.empty:
        raise ValueError(f"Нет истории для сегмента {segment!r} и категории {category!r}.")
    with closing(_connect()) as connection:
        bundle = _load_active_bundle(connection)
    if bundle is None:
        raise RuntimeError("Нет активной модели для объяснения прогноза.")
    row = selected.iloc[0][FEATURE_COLUMNS].to_dict()
    baseline_values = {
        "segment": str(daily["segment"].mode().iloc[0]),
        "category": str(daily["category"].mode().iloc[0]),
        **{
            feature: float(daily[feature].median())
            for feature in NUMERIC_FEATURES
        },
    }
    original_x = _encoded_forecasts(pd.DataFrame([row]), bundle["columns"])
    forecast = float(_predict(bundle["model_name"], bundle["model"], original_x)[0])
    explanations: list[dict[str, object]] = []
    for feature in FEATURE_COLUMNS:
        reference_row = dict(row)
        reference_row[feature] = baseline_values[feature]
        reference_x = _encoded_forecasts(pd.DataFrame([reference_row]), bundle["columns"])
        reference_prediction = float(
            _predict(bundle["model_name"], bundle["model"], reference_x)[0]
        )
        explanations.append(
            {
                "Признак": feature,
                "Значение": str(row[feature]),
                "Типичное значение": str(baseline_values[feature]),
                "Изменение прогноза, п.п.": (forecast - reference_prediction) * 100,
            }
        )
    return forecast, pd.DataFrame(explanations).sort_values(
        "Изменение прогноза, п.п.", key=lambda values: values.abs(), ascending=False
    ).reset_index(drop=True)


def feature_dependence(
    events: pd.DataFrame,
    segment: str,
    category: str,
    feature: str,
    forecast_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Vary one input feature while holding the selected forecast's other inputs fixed."""
    if feature not in FEATURE_COLUMNS:
        raise ValueError(f"Неизвестный признак для анализа зависимости: {feature}")
    daily = _daily_data(events, include_unlabeled=True)
    rows = daily.loc[
        (daily["segment"] == segment) & (daily["category"] == category)
    ].sort_values("date")
    if forecast_date is None:
        selected = rows.tail(1)
    else:
        requested_date = pd.Timestamp(forecast_date).normalize()
        selected = rows.loc[rows["target_date"] == requested_date]
    if selected.empty:
        raise ValueError(
            f"Нет прогноза для {segment!r}/{category!r} на указанную дату."
        )
    with closing(_connect()) as connection:
        bundle = _load_active_bundle(connection)
    if bundle is None:
        raise RuntimeError("Нет активной модели для анализа зависимости признака.")
    base = selected.iloc[0][FEATURE_COLUMNS].to_dict()
    if feature == "segment":
        values: list[Any] = sorted(daily["segment"].dropna().astype(str).unique())
    elif feature == "category":
        values = sorted(daily["category"].dropna().astype(str).unique())
    elif feature in {"target_weekday", "target_weekend"}:
        values = list(range(7)) if feature == "target_weekday" else [0, 1]
    else:
        values = sorted(
            set(
                float(value)
                for value in daily[feature].quantile(np.linspace(0, 1, 11)).dropna()
            )
        )
    varied_rows = []
    for value in values:
        varied = dict(base)
        varied[feature] = value
        if feature == "target_weekday":
            varied["target_weekend"] = int(value in [5, 6])
        elif feature == "target_weekend":
            varied["target_weekday"] = 5 if value else 2
        varied_rows.append(varied)
    features = _encoded_forecasts(pd.DataFrame(varied_rows), bundle["columns"])
    predictions = _predict(bundle["model_name"], bundle["model"], features)
    return pd.DataFrame(
        {
            "Значение": [row[feature] for row in varied_rows],
            "Прогноз CTR": predictions,
            "Прогноз кликов на 1000": predictions * 1000,
        }
    )


def analyze_feature_importance(events: pd.DataFrame) -> pd.DataFrame:
    """Compare permutation importance between validation and later holdout periods."""
    daily = _daily_data(events)
    encoded = _encoded_features(daily)
    validation_start, holdout_start = _split_dates(daily)
    train_mask = daily["target_date"] < validation_start
    validation_mask = (daily["target_date"] >= validation_start) & (
        daily["target_date"] < holdout_start
    )
    holdout_mask = daily["target_date"] >= holdout_start
    with closing(_connect()) as connection:
        bundle = _load_active_bundle(connection)
    if bundle is None:
        raise RuntimeError("Нет активной модели для анализа признаков.")

    evaluator = _new_model(bundle["model_name"], bundle["parameters"], bundle["seed"])
    train_x = encoded.loc[train_mask].reindex(columns=bundle["columns"], fill_value=0)
    train_y = daily.loc[train_mask, "target"]
    if evaluator is not None:
        evaluator.fit(train_x, train_y)
    grouped_values: dict[tuple[str, str], float] = {}
    grouped_information: dict[tuple[str, str], float] = {}
    for period, mask in (("Validation", validation_mask), ("Holdout", holdout_mask)):
        period_x = encoded.loc[mask].reindex(columns=bundle["columns"], fill_value=0)
        period_y = daily.loc[mask, "target"]
        if len(period_x) < 2:
            continue
        if bundle["model_name"] == "Наивная (CTR вчера)":
            baseline_prediction = _predict(bundle["model_name"], None, period_x)
            base_mae = mean_absolute_error(period_y, baseline_prediction)
            permutation_scores: dict[str, float] = {}
            for column in period_x.columns:
                shuffled = period_x.copy()
                shuffled[column] = shuffled[column].sample(
                    frac=1, random_state=bundle["seed"]
                ).to_numpy()
                shuffled_prediction = _predict(bundle["model_name"], None, shuffled)
                permutation_scores[column] = max(
                    0.0, float(mean_absolute_error(period_y, shuffled_prediction) - base_mae)
                )
        else:
            permutation = permutation_importance(
                evaluator,
                period_x,
                period_y,
                scoring="neg_mean_absolute_error",
                n_repeats=5,
                random_state=bundle["seed"],
                n_jobs=1,
            )
            permutation_scores = {
                column: max(0.0, float(score))
                for column, score in zip(period_x.columns, permutation.importances_mean)
            }
        mutual_information = mutual_info_regression(
            period_x,
            period_y,
            random_state=bundle["seed"],
        )
        information_scores = {
            column: max(0.0, float(score))
            for column, score in zip(period_x.columns, mutual_information)
        }
        for column, score in permutation_scores.items():
            if column.startswith("segment_"):
                feature = "segment"
            elif column.startswith("category_"):
                feature = "category"
            else:
                feature = column
            key = (feature, period)
            grouped_values[key] = grouped_values.get(key, 0.0) + score
            grouped_information[key] = grouped_information.get(key, 0.0) + information_scores[column]

    rows = [
        {
            "Признак": feature,
            "Период": period,
            "Permutation importance (ΔMAE)": value,
            "Mutual information (nats)": grouped_information.get((feature, period), 0.0),
        }
        for (feature, period), value in grouped_values.items()
    ]
    importance = pd.DataFrame(rows)
    if bundle["model_name"] == "Случайный лес":
        intrinsic = dict(zip(bundle["columns"], bundle["model"].feature_importances_))
        grouped_intrinsic: dict[str, float] = {}
        for column, value in intrinsic.items():
            feature = "segment" if column.startswith("segment_") else (
                "category" if column.startswith("category_") else column
            )
            grouped_intrinsic[feature] = grouped_intrinsic.get(feature, 0.0) + float(value)
        importance["Встроенная важность (Gini)"] = importance["Признак"].map(grouped_intrinsic)
    else:
        importance["Встроенная важность (|коэффициент|)"] = np.nan
        if bundle["model_name"] == "Линейная регрессия":
            grouped_intrinsic = {}
            for column, value in zip(bundle["columns"], np.abs(bundle["model"].coef_)):
                feature = "segment" if column.startswith("segment_") else (
                    "category" if column.startswith("category_") else column
                )
                grouped_intrinsic[feature] = grouped_intrinsic.get(feature, 0.0) + float(value)
            importance["Встроенная важность (|коэффициент|)"] = importance["Признак"].map(grouped_intrinsic)
    return importance.sort_values(
        ["Период", "Permutation importance (ΔMAE)"], ascending=[True, False]
    ).reset_index(drop=True)


def _population_stability_index(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 5 or len(actual) < 5:
        return 0.0
    unique_values = np.unique(np.concatenate((expected, actual)))
    if len(unique_values) <= 10:
        midpoints = (unique_values[:-1] + unique_values[1:]) / 2
        edges = np.concatenate(([-np.inf], midpoints, [np.inf]))
    else:
        quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, 11)))
        if len(quantiles) < 3:
            return 0.0
        edges = np.concatenate(([-np.inf], quantiles[1:-1], [np.inf]))
    expected_counts = np.histogram(expected, bins=edges)[0].astype(float)
    actual_counts = np.histogram(actual, bins=edges)[0].astype(float)
    expected_share = (expected_counts + 1e-4) / (expected_counts.sum() + 1e-4 * len(expected_counts))
    actual_share = (actual_counts + 1e-4) / (actual_counts.sum() + 1e-4 * len(actual_counts))
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def monitor_model(
    events: pd.DataFrame, psi_threshold: float = 0.20, error_multiplier: float = 1.25
) -> tuple[pd.DataFrame, bool]:
    """Persist drift and recent error diagnostics; return whether retraining is advised."""
    daily = _daily_data(events)
    with closing(_connect()) as connection, connection:
        row = _active_model_row(connection)
        bundle = _load_active_bundle(connection)
        if row is None or bundle is None:
            raise RuntimeError("Сначала обучите и зарегистрируйте модель.")
        reference = pd.DataFrame(bundle["reference"])
        latest_date = pd.Timestamp(daily["date"].max())
        recent_start = latest_date - pd.Timedelta(days=13)
        prior_start = latest_date - pd.Timedelta(days=27)
        same_training_data = _dataset_hash(events) == bundle["data_hash"]
        if same_training_data and bundle.get("holdout_model") is not None:
            recent = daily.loc[
                daily["target_date"] >= pd.Timestamp(bundle["holdout_start"])
            ]
            scoring_model = bundle["holdout_model"]
        else:
            recent = daily.loc[daily["date"] >= recent_start]
            scoring_model = bundle["model"]
        previous = daily.loc[
            (daily["date"] >= prior_start) & (daily["date"] < recent_start)
        ]
        recent_features = _encoded_features(recent)
        drift_features = [
            feature
            for feature in reference.columns
            if feature in NUMERIC_FEATURES
            or feature.startswith(("segment_", "category_"))
        ]
        psi_by_feature = {
            feature: _population_stability_index(
                reference[feature].to_numpy(dtype=float),
                recent_features.reindex(columns=[feature], fill_value=0)[feature].to_numpy(
                    dtype=float
                ),
            )
            for feature in drift_features
            if feature in reference
        }
        maximum_psi = max(psi_by_feature.values(), default=0.0)
        if recent.empty:
            recent_mae = float("nan")
        else:
            recent_x = _encoded_features(recent).reindex(
                columns=bundle["columns"], fill_value=0
            )
            recent_prediction = _predict(bundle["model_name"], scoring_model, recent_x)
            recent_mae = float(mean_absolute_error(recent["target"], recent_prediction))
        baseline_mae = float(row["holdout_mae"])
        drift_detected = maximum_psi >= psi_threshold
        error_degraded = np.isfinite(recent_mae) and recent_mae > baseline_mae * error_multiplier
        retrain_recommended = bool(drift_detected or error_degraded)
        details = {
            "psi_threshold": psi_threshold,
            "error_multiplier": error_multiplier,
            "psi_by_feature": psi_by_feature,
            "error_degraded": bool(error_degraded),
            "window_dates": {
                "from": str(recent["date"].min()) if not recent.empty else None,
                "to": str(recent["date"].max()) if not recent.empty else None,
                "previous_window_from": str(previous["date"].min()) if not previous.empty else None,
            },
        }
        connection.execute(
            "INSERT INTO monitoring_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                datetime.now(timezone.utc).isoformat(),
                row["version_id"],
                maximum_psi,
                recent_mae,
                baseline_mae,
                int(drift_detected),
                int(retrain_recommended),
                json.dumps(details, sort_keys=True),
            ),
        )
    report = pd.DataFrame(
        [
            {"Показатель": "Максимальный PSI", "Значение": maximum_psi, "Порог": psi_threshold},
            {"Показатель": "MAE последнего периода", "Значение": recent_mae, "Порог": baseline_mae * error_multiplier},
            {"Показатель": "MAE holdout при обучении", "Значение": baseline_mae, "Порог": np.nan},
        ]
    )
    return report, retrain_recommended


def get_model_registry() -> pd.DataFrame:
    with closing(_connect()) as connection:
        return pd.read_sql_query(
            """SELECT version_id AS "Версия", created_at AS "Создана",
                      model_name AS "Модель", parameters AS "Параметры",
                      validation_mae AS "Validation MAE",
                      holdout_mae AS "Holdout MAE", status AS "Статус"
               FROM model_registry ORDER BY created_at DESC""",
            connection,
        )


def get_experiment_log() -> pd.DataFrame:
    with closing(_connect()) as connection:
        return pd.read_sql_query(
            """SELECT experiment_id AS "Эксперимент", created_at AS "Создан",
                      model_name AS "Модель", parameters AS "Параметры",
                      seed AS "Seed", data_hash AS "Хэш данных",
                      train_rows AS "Строк обучения",
                      validation_mae AS "Validation MAE",
                      validation_rmse AS "Validation RMSE",
                      selected AS "Выбран"
               FROM experiments ORDER BY created_at DESC LIMIT 200""",
            connection,
        )


def replay_experiment(experiment_id: str) -> pd.DataFrame:
    """Re-run a logged configuration against its immutable event snapshot."""
    with closing(_connect()) as connection:
        experiment = connection.execute(
            """SELECT model_name, parameters, seed, data_hash
               FROM experiments WHERE experiment_id=?""",
            (experiment_id,),
        ).fetchone()
        if experiment is None:
            raise ValueError(f"Эксперимент не найден: {experiment_id}")
        snapshot = connection.execute(
            "SELECT events_gzip FROM dataset_snapshots WHERE data_hash=?",
            (experiment[3],),
        ).fetchone()
    if snapshot is None:
        raise RuntimeError(f"Снимок данных {experiment[3]} не найден.")
    events = pd.read_csv(
        io.BytesIO(gzip.decompress(snapshot[0])),
        parse_dates=["timestamp"],
    )
    if _dataset_hash(events) != experiment[3]:
        raise RuntimeError("Контрольная сумма снимка данных не совпадает с экспериментом.")

    daily = _daily_data(events)
    encoded = _encoded_features(daily)
    validation_start, holdout_start = _split_dates(daily)
    train_mask = daily["target_date"] < validation_start
    validation_mask = (daily["target_date"] >= validation_start) & (
        daily["target_date"] < holdout_start
    )
    holdout_mask = daily["target_date"] >= holdout_start
    model_name = str(experiment[0])
    parameters = json.loads(experiment[1])
    feature_columns = parameters.pop("features", list(encoded.columns))
    parameters.pop("feature_set", None)
    seed = int(experiment[2])
    model = _new_model(model_name, parameters, seed)
    if model is not None:
        model.fit(
            encoded.loc[train_mask].reindex(columns=feature_columns, fill_value=0),
            daily.loc[train_mask, "target"],
        )
    validation_prediction = _predict(
        model_name,
        model,
        encoded.loc[validation_mask].reindex(columns=feature_columns, fill_value=0),
    )
    validation_target = daily.loc[validation_mask, "target"]
    holdout_model = _new_model(model_name, parameters, seed)
    if holdout_model is not None:
        holdout_model.fit(
            pd.concat([encoded.loc[train_mask], encoded.loc[validation_mask]])
            .reindex(columns=feature_columns, fill_value=0),
            pd.concat([daily.loc[train_mask, "target"], daily.loc[validation_mask, "target"]]),
        )
    holdout_prediction = _predict(
        model_name,
        holdout_model,
        encoded.loc[holdout_mask].reindex(columns=feature_columns, fill_value=0),
    )
    holdout_target = daily.loc[holdout_mask, "target"]
    validation_mae = float(mean_absolute_error(validation_target, validation_prediction))
    validation_rmse = float(np.sqrt(mean_squared_error(validation_target, validation_prediction)))
    holdout_mae = float(mean_absolute_error(holdout_target, holdout_prediction))
    holdout_rmse = float(np.sqrt(mean_squared_error(holdout_target, holdout_prediction)))
    with closing(_connect()) as connection, connection:
        connection.execute(
            "INSERT INTO experiment_replays VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                experiment_id,
                datetime.now(timezone.utc).isoformat(),
                str(experiment[3]),
                validation_mae,
                validation_rmse,
                holdout_mae,
                holdout_rmse,
            ),
        )
    return pd.DataFrame(
        [
            {"Период": "Validation", "MAE": validation_mae, "RMSE": validation_rmse},
            {"Период": "Holdout", "MAE": holdout_mae, "RMSE": holdout_rmse},
        ]
    )


def get_experiment_replays() -> pd.DataFrame:
    with closing(_connect()) as connection:
        return pd.read_sql_query(
            """SELECT original_experiment_id AS "Эксперимент",
                      replayed_at AS "Повторный запуск",
                      validation_mae AS "Validation MAE",
                      validation_rmse AS "Validation RMSE",
                      holdout_mae AS "Holdout MAE",
                      holdout_rmse AS "Holdout RMSE"
               FROM experiment_replays ORDER BY replayed_at DESC LIMIT 100""",
            connection,
        )


def get_monitoring_log() -> pd.DataFrame:
    with closing(_connect()) as connection:
        return pd.read_sql_query(
            """SELECT created_at AS "Проверка", version_id AS "Версия",
                      maximum_psi AS "Максимальный PSI",
                      recent_mae AS "MAE последнего периода",
                      baseline_mae AS "MAE holdout",
                      drift_detected AS "Обнаружен дрейф",
                      retrain_recommended AS "Нужно переобучение",
                      details AS "Диагностика"
               FROM monitoring_log ORDER BY created_at DESC LIMIT 100""",
            connection,
        )


def get_quality_log() -> pd.DataFrame:
    with closing(_connect()) as connection:
        return pd.read_sql_query(
            """SELECT checked_at AS "Проверка", severity AS "Результат",
                      check_name AS "Контроль", details AS "Подробности"
               FROM quality_log ORDER BY checked_at DESC LIMIT 200""",
            connection,
        )


def read_feature_store(data_hash: str) -> pd.DataFrame:
    with closing(_connect()) as connection:
        return pd.read_sql_query(
            """SELECT date, segment, category, impressions, clicks, ctr,
                      ctr_lag_1, ctr_mean_7, impressions_lag_1, target,
                      target_date, target_weekday, target_weekend
               FROM feature_store WHERE data_hash=?
               ORDER BY date, segment, category""",
            connection,
            params=(data_hash,),
        )


def activate_model_version(version_id: str) -> None:
    with closing(_connect()) as connection, connection:
        row = connection.execute(
            "SELECT version_id FROM model_registry WHERE version_id = ?", (version_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Версия модели не найдена: {version_id}")
        connection.execute("UPDATE model_registry SET status='archived'")
        connection.execute(
            "UPDATE model_registry SET status='active' WHERE version_id=?", (version_id,)
        )
        connection.execute(
            """INSERT INTO active_model(singleton, version_id) VALUES (1, ?)
               ON CONFLICT(singleton) DO UPDATE SET version_id=excluded.version_id""",
            (version_id,),
        )


def rollback_model() -> str:
    with closing(_connect()) as connection, connection:
        active = connection.execute(
            "SELECT version_id FROM active_model WHERE singleton=1"
        ).fetchone()
        if active is None:
            raise RuntimeError("Нет активной версии модели для отката.")
        prior = connection.execute(
            """SELECT version_id FROM model_registry
               WHERE version_id != ? ORDER BY created_at DESC LIMIT 1""",
            (active[0],),
        ).fetchone()
        if prior is None:
            raise RuntimeError("Нет предыдущей версии модели для отката.")
        connection.execute("UPDATE model_registry SET status='archived'")
        connection.execute(
            "UPDATE model_registry SET status='active' WHERE version_id=?", (prior[0],)
        )
        connection.execute(
            """INSERT INTO active_model(singleton, version_id) VALUES (1, ?)
               ON CONFLICT(singleton) DO UPDATE SET version_id=excluded.version_id""",
            (prior[0],),
        )
        return str(prior[0])


def recommend_ads(
    user_id: str, events: pd.DataFrame, ads: pd.DataFrame, forecasts: pd.DataFrame
) -> pd.DataFrame:
    history = events.loc[events["user_id"].astype(str) == str(user_id)]
    if history.empty:
        return ads.assign(
            score=0.0, forecast_ctr=0.0, forecast_clicks_per_1000=0.0
        )
    segment = str(history["segment"].mode().iloc[0])
    preferences = (
        history.groupby("category")
        .agg(
            views=("viewed", "sum"),
            clicks=("clicked", "sum"),
            likes=("liked", "sum"),
            reposts=("reposted", "sum"),
        )
    )
    preferences["interest"] = (
        preferences["views"]
        + 2 * preferences["clicks"]
        + 3 * preferences["likes"]
        + 4 * preferences["reposts"]
    )
    preferences["interest"] /= max(float(preferences["interest"].max()), 1.0)
    segment_forecasts = forecasts.loc[
        forecasts["segment"] == segment, ["category", "forecast_ctr"]
    ]
    ranked = ads.merge(segment_forecasts, on="category", how="left").merge(
        preferences[["interest"]], left_on="category", right_index=True, how="left"
    )
    ranked["interest"] = ranked["interest"].fillna(0)
    ranked["forecast_ctr"] = ranked["forecast_ctr"].fillna(0)
    ranked["score"] = ranked["forecast_ctr"] * (0.35 + 0.65 * ranked["interest"])
    ranked["forecast_clicks_per_1000"] = ranked["forecast_ctr"] * 1000
    return ranked.sort_values("score", ascending=False).reset_index(drop=True)
