from pathlib import Path

TEST_DATABASE_PATH = Path(__file__).parents[2].joinpath("tests", "data", "bear.db")
TEST_DATABASE_URL = f"sqlite:///{TEST_DATABASE_PATH}"
