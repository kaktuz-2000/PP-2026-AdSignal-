import tempfile
import unittest
from pathlib import Path

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
                recommender.load_sources(events, ads)
                recommender.load_sources(events, ads)
                loaded_events, loaded_ads, _ = recommender.read_sources()
                self.assertEqual(len(loaded_events), len(events))
                self.assertEqual(len(loaded_ads), len(ads))

                metrics, forecasts, _ = recommender.train_models(loaded_events)
                ranked_ads = recommender.recommend_ads(
                    "user_000", loaded_events, loaded_ads, forecasts
                )
                self.assertEqual(len(metrics), 3)
                self.assertEqual(len(forecasts), 20)
                self.assertFalse(ranked_ads.empty)
            finally:
                recommender.RAW_DIR = original_raw_dir
                recommender.DB_PATH = original_db_path


if __name__ == "__main__":
    unittest.main()