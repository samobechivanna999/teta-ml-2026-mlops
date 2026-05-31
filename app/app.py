import json
import logging
import os
import sys
import time
from datetime import datetime

import joblib
import pandas as pd
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

sys.path.append(os.path.abspath("./src"))

from preprocessing import run_preproc
from scorer import make_pred, save_feature_importance_top5, save_score_density_plot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/app/logs/service.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ProcessingService:
    def __init__(self) -> None:
        logger.info("Initializing ProcessingService...")
        self.input_dir = "/app/input"
        self.output_dir = "/app/output"
        bundle = joblib.load("./models/model_bundle.joblib")
        self.artifacts = bundle["artifacts"]
        logger.info("Model and preprocessing artifacts loaded")

    def process_single_file(self, file_path: str) -> None:
        try:
            logger.info("Processing file: %s", file_path)
            input_df = pd.read_csv(file_path)

            logger.info("Starting preprocessing")
            processed_df = run_preproc(input_df, self.artifacts)

            logger.info("Making prediction")
            submission, scores = make_pred(processed_df, file_path)

            base_name = os.path.splitext(os.path.basename(file_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = f"{base_name}_{timestamp}"

            submission_path = os.path.join(self.output_dir, f"sample_submission_{prefix}.csv")
            submission.to_csv(submission_path, index=False)
            logger.info("Submission saved to: %s", submission_path)

            fi_path = os.path.join(self.output_dir, f"feature_importance_top5_{prefix}.json")
            save_feature_importance_top5(fi_path)

            plot_path = os.path.join(self.output_dir, f"score_density_{prefix}.png")
            save_score_density_plot(scores, plot_path)

            meta_path = os.path.join(self.output_dir, f"run_meta_{prefix}.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "input_file": os.path.basename(file_path),
                        "rows_scored": int(len(submission)),
                        "prediction_min": float(scores.min()),
                        "prediction_max": float(scores.max()),
                        "outputs": {
                            "submission": os.path.basename(submission_path),
                            "feature_importance": os.path.basename(fi_path),
                            "density_plot": os.path.basename(plot_path),
                        },
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info("Scoring pipeline finished successfully")

        except Exception as exc:
            logger.error("Error processing file %s: %s", file_path, exc, exc_info=True)


class FileHandler(FileSystemEventHandler):
    def __init__(self, service: ProcessingService) -> None:
        self.service = service

    def on_created(self, event) -> None:
        if not event.is_directory and event.src_path.endswith(".csv"):
            logger.debug("New file detected: %s", event.src_path)
            self.service.process_single_file(event.src_path)


if __name__ == "__main__":
    logger.info("Starting ML scoring service...")
    service = ProcessingService()
    observer = Observer()
    observer.schedule(FileHandler(service), path=service.input_dir, recursive=False)
    observer.start()
    logger.info("File observer started. Put test.csv into ./input")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
        observer.stop()
        observer.join()
