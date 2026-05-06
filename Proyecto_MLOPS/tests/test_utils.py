import pytest
from src.utils import load_data, prepare_data

DATA_PATH = "data/features_30_sec.csv"


@pytest.fixture(scope="session")
def raw_dataframe():
    df = load_data(DATA_PATH)
    return df


def test_raw_dataset_not_empty(raw_dataframe):
    assert len(raw_dataframe) > 0


def test_raw_dataset_has_target_column(raw_dataframe):
    assert ("label" in raw_dataframe.columns) or ("genre" in raw_dataframe.columns)


def test_prepare_data_structure():
    data = prepare_data(DATA_PATH, seed=42)

    required_keys = [
        "X_train", "X_val", "X_test",
        "y_train", "y_val", "y_test",
        "input_dim", "num_classes"
    ]

    for key in required_keys:
        assert key in data