"""Script entrypoint for full pipeline execution."""

from pipeline.scheduler import run_full_pipeline


if __name__ == "__main__":
    result = run_full_pipeline(force_retrain=False)
    print(result)
