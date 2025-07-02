"""This file contains code to create features for modeling, i.e. data which wasn't in the raw data but which is being made available for the model.
It shall never output to "raw" data.  It is meant to take raw or interim data as input and deliver interim data as output.

Examples of data feature-generation include:

Given a vacancy is on record as having existed in a specified court, with a named incumbent judge, a known reason the vacancy occurred, and a known date the vacancy began, when a vacancy with equivalent data is found on other months, then it shall be treated as the same vacancy incident.
Given earlier records for a vacancy incident lack a nominee and nomination date, when a later record for the same vacancy incident has a nominee and nomination date, then the nominee and nomination date shall be merged onto that vacancy incident's records in interim data.
Given a vacancy is on record as having existed in a specified court, with a named incumbent judge, when a vacancy with equivalent court location and incumbent is found to have occurred with a different reason and/or vacancy date (e.g. a nomination is withdrawn and reopened), then the two vacancy incidents shall be treated as each deserving of their own unique record in interim data.
Given a nomination is on record as having occurred on a specified date, when filling in date-dependent feature data (e.g. which President performed the nomination), then the interim data shall be updated to include date-inferred data (e.g. a numeric ordinal indicator identifying that President who was in office on the date of nomination.
"""

from loguru import logger
from tqdm import tqdm
import typer

from nomination_predictor.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
