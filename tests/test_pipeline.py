import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src import ad_recommender as recommender


class PipelineTests(unittest.TestCase):
    def test_ingestion_is_idempotent_and_models_produce_recommendations(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            original_raw_dir = recommender.RAW_DIR
            original_db_path = recommender.DB_PATH
            recommender.RAW_DIR = root / "raw"
            recommender.DB_PATH = root / "processed" / "ads.sqlite3"
            try:
                events, ads = recommender.create_sources(days=21)
                quality = recommender.validate_sources(events, ads)
                self.assertFalse((quality["Результат"] == "ERROR").any())
                original_features = recommender._daily_data(events)
                altered_events = events.copy()
                last_date = pd.to_datetime(altered_events["timestamp"]).max().floor("D")
                last_day = pd.to_datetime(altered_events["timestamp"]).dt.floor("D") == last_date
                altered_events.loc[last_day, "clicked"] = 0
                changed_features = recommender._daily_data(altered_events)
                comparison_columns = [
                    "date", "segment", "category", *recommender.FEATURE_COLUMNS
                ]
                pd.testing.assert_frame_equal(
                    original_features.loc[original_features["date"] < last_date, comparison_columns]
                    .reset_index(drop=True),
                    changed_features.loc[changed_features["date"] < last_date, comparison_columns]
                    .reset_index(drop=True),
                )
                recommender.load_sources(events, ads)
                recommender.load_sources(events, ads)
                loaded_events, loaded_ads, _ = recommender.read_sources()
                self.assertEqual(len(loaded_events), len(events))
                self.assertEqual(len(loaded_ads), len(ads))

                metrics, forecasts, best_model = recommender.train_models(
                    loaded_events, search_budget=2
                )
                ranked_ads = recommender.recommend_ads(
                    "user_000", loaded_events, loaded_ads, forecasts
                )
                self.assertEqual(len(metrics), 3)
                self.assertEqual(len(forecasts), 20)
                stored_features = recommender.read_feature_store(
                    recommender.data_fingerprint(loaded_events)
                )
                self.assertFalse(stored_features.empty)
                last_feature_date = pd.to_datetime(stored_features["date"]).max()
                self.assertTrue(
                    stored_features.loc[
                        pd.to_datetime(stored_features["date"]) == last_feature_date,
                        "target",
                    ].isna().all()
                )
                self.assertEqual(
                    pd.to_datetime(forecasts["forecast_date"]).max(),
                    pd.to_datetime(loaded_events["timestamp"]).max().floor("D")
                    + pd.Timedelta(days=1),
                )
                self.assertEqual(int(metrics["Выбрана по validation"].sum()), 1)
                self.assertIn("Набор признаков", metrics.columns)
                self.assertIn(best_model, metrics["Модель"].tolist())
                self.assertFalse(ranked_ads.empty)
                numeric_user_events = loaded_events.copy()
                numeric_user_events["user_id"] = (
                    numeric_user_events["user_id"].str.extract(r"(\d+)")[0].astype(int)
                )
                numeric_user_recommendations = recommender.recommend_ads(
                    "0", numeric_user_events, loaded_ads, forecasts
                )
                self.assertGreater(
                    float(numeric_user_recommendations["score"].max()), 0.0
                )

                explanation_ctr, explanation = recommender.explain_forecast(
                    loaded_events,
                    "student",
                    "technology",
                    forecast_date=pd.to_datetime(loaded_events["timestamp"]).max().floor("D")
                    + pd.Timedelta(days=1),
                )
                self.assertGreaterEqual(explanation_ctr, 0)
                self.assertEqual(len(explanation), len(recommender.FEATURE_COLUMNS))
                dependence = recommender.feature_dependence(
                    loaded_events,
                    "student",
                    "technology",
                    "ctr_lag_1",
                )
                self.assertGreaterEqual(len(dependence), 2)

                importance = recommender.analyze_feature_importance(loaded_events)
                self.assertIn("Permutation importance (ΔMAE)", importance.columns)
                self.assertIn("Mutual information (nats)", importance.columns)
                self.assertEqual(set(importance["Период"]), {"Validation", "Holdout"})

                report, retrain_recommended = recommender.monitor_model(loaded_events)
                self.assertFalse(report.empty)
                self.assertIsInstance(retrain_recommended, bool)

                experiments = recommender.get_experiment_log()
                self.assertGreaterEqual(len(experiments), 4)
                experiment = experiments.iloc[0]
                replay = recommender.replay_experiment(str(experiment["Эксперимент"]))
                self.assertEqual(set(replay["Период"]), {"Validation", "Holdout"})
                self.assertAlmostEqual(
                    float(experiment["Validation MAE"]),
                    float(replay.loc[replay["Период"] == "Validation", "MAE"].iloc[0]),
                    places=10,
                )
                registry = recommender.get_model_registry()
                self.assertEqual(int((registry["Статус"] == "active").sum()), 1)
                with self.assertRaises(RuntimeError):
                    recommender.rollback_model()

                shifted_events, shifted_ads = recommender.create_sources(
                    seed=43, days=42, drift=True
                )
                recommender.load_sources(shifted_events, shifted_ads, replace=True)
                drift_report, drift_retrain = recommender.monitor_model(shifted_events)
                self.assertTrue(drift_retrain, drift_report.to_string(index=False))
                previous_version = str(
                    recommender.get_model_registry()
                    .loc[lambda frame: frame["Статус"] == "active", "Версия"]
                    .iloc[0]
                )
                recommender.train_models(
                    shifted_events, search_budget=2, promotion_threshold=0
                )
                promoted_registry = recommender.get_model_registry()
                self.assertGreaterEqual(len(promoted_registry), 2)
                self.assertNotEqual(
                    str(
                        promoted_registry.loc[
                            promoted_registry["Статус"] == "active", "Версия"
                        ].iloc[0]
                    ),
                    previous_version,
                )
                self.assertEqual(recommender.rollback_model(), previous_version)

                invalid_events = events.copy()
                invalid_events.loc[0, "clicked"] = 2
                with self.assertRaises(ValueError):
                    recommender.load_sources(invalid_events, ads)
            finally:
                recommender.RAW_DIR = original_raw_dir
                recommender.DB_PATH = original_db_path


if __name__ == "__main__":
    unittest.main()