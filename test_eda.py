"""
Tests the functions in eda.py.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import eda


def make_small_data() -> pd.DataFrame:
    """
    Creates a small dataset with manually verifiable values.
    """
    return pd.DataFrame({
        "label": [0, 1],
        "pixel1": [0, 10],
        "pixel2": [20, 10],
        "pixel3": [0, 10],
        "pixel4": [20, 10],
    })


def make_fashion_data() -> pd.DataFrame:
    """
    Creates a small dataset with 784 pixel columns and all ten labels.
    """
    data = {
        "label": list(range(10))
    }

    for pixel_number in range(784):
        column_name = f"pixel{pixel_number + 1}"
        data[column_name] = [
            label * 10
            for label in range(10)
        ]

    return pd.DataFrame(data)


def test_get_pixel_columns() -> None:
    """
    Tests that the label column is excluded from the pixel columns.
    """
    data = make_small_data()

    actual = eda.get_pixel_columns(data)

    expected = [
        "pixel1",
        "pixel2",
        "pixel3",
        "pixel4",
    ]

    assert actual == expected


def test_count_missing() -> None:
    """
    Tests counting missing values in a dataset.
    """
    data = pd.DataFrame({
        "label": [0, 1, 2],
        "pixel1": [0, None, 20],
        "pixel2": [None, 100, 200],
    })

    assert eda.count_missing(data) == 2


def test_count_no_missing() -> None:
    """
    Tests a dataset containing no missing values.
    """
    data = make_small_data()

    assert eda.count_missing(data) == 0


def test_create_image_summary() -> None:
    """
    Tests the calculated image-level summary variables.
    """
    data = make_small_data()

    result = eda.create_image_summary(data)

    assert result.loc[0, "label"] == 0
    assert result.loc[1, "label"] == 1

    assert result.loc[0, "category"] == "T-shirt/top"
    assert result.loc[1, "category"] == "Trouser"

    assert result.loc[0, "mean_intensity"] == 10
    assert result.loc[1, "mean_intensity"] == 10

    assert result.loc[0, "nonzero_pixels"] == 2
    assert result.loc[1, "nonzero_pixels"] == 4

    assert result.loc[1, "pixel_std"] == 0


def test_validate_dataset() -> None:
    """
    Tests a correctly structured Fashion-MNIST dataset.
    """
    data = make_fashion_data()

    eda.validate_dataset(data, "Test")


def test_validate_missing_label() -> None:
    """
    Tests that validation rejects data without a label column.
    """
    data = make_fashion_data()
    data = data.drop(columns="label")

    error_found = False

    try:
        eda.validate_dataset(data, "Test")
    except ValueError:
        error_found = True

    assert error_found


def test_validate_wrong_pixel_count() -> None:
    """
    Tests that validation rejects an incorrect number of pixels.
    """
    data = make_small_data()
    error_found = False

    try:
        eda.validate_dataset(data, "Test")
    except ValueError:
        error_found = True

    assert error_found


def test_validate_invalid_label() -> None:
    """
    Tests that validation rejects labels outside zero through nine.
    """
    data = make_fashion_data()
    data.loc[0, "label"] = 10
    error_found = False

    try:
        eda.validate_dataset(data, "Test")
    except ValueError:
        error_found = True

    assert error_found


def test_load_data() -> None:
    """
    Tests loading training and testing data from CSV files.
    """
    expected_train = make_small_data()
    expected_test = make_small_data()

    with TemporaryDirectory() as directory:
        directory_path = Path(directory)
        train_path = directory_path / "train.csv"
        test_path = directory_path / "test.csv"

        expected_train.to_csv(
            train_path,
            index=False
        )
        expected_test.to_csv(
            test_path,
            index=False
        )

        actual_train, actual_test = eda.load_data(
            train_path,
            test_path
        )

        assert actual_train.equals(expected_train)
        assert actual_test.equals(expected_test)


def test_load_missing_file() -> None:
    """
    Tests that loading a nonexistent file raises an error.
    """
    with TemporaryDirectory() as directory:
        directory_path = Path(directory)
        missing_train = directory_path / "missing_train.csv"
        missing_test = directory_path / "missing_test.csv"
        error_found = False

        try:
            eda.load_data(
                missing_train,
                missing_test
            )
        except FileNotFoundError:
            error_found = True

        assert error_found


def test_visualizations() -> None:
    """
    Tests that each visualization creates an image file.
    """
    fashion_data = make_fashion_data()
    image_summary = eda.create_image_summary(
        fashion_data
    )

    original_directory = eda.RESULTS_DIRECTORY

    with TemporaryDirectory() as directory:
        temporary_directory = Path(directory)
        eda.RESULTS_DIRECTORY = temporary_directory

        try:
            eda.plot_class_counts(image_summary)
            eda.plot_intensity_by_category(image_summary)
            eda.plot_average_images(fashion_data)

            assert (
                temporary_directory
                / "class_counts.png"
            ).exists()

            assert (
                temporary_directory
                / "intensity_by_category.png"
            ).exists()

            assert (
                temporary_directory
                / "average_images.png"
            ).exists()
        finally:
            eda.RESULTS_DIRECTORY = original_directory


def main() -> None:
    """
    Runs all tests.
    """
    test_get_pixel_columns()
    test_count_missing()
    test_count_no_missing()
    test_create_image_summary()
    test_validate_dataset()
    test_validate_missing_label()
    test_validate_wrong_pixel_count()
    test_validate_invalid_label()
    test_load_data()
    test_load_missing_file()
    test_visualizations()

    print("All tests passed!")


if __name__ == "__main__":
    main()
