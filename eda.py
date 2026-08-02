"""
Performs exploratory data analysis on the Fashion-MNIST dataset.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIRECTORY = Path(__file__).resolve().parent
TRAIN_PATH = PROJECT_DIRECTORY / "data" / "fashion-mnist_train.csv"
TEST_PATH = PROJECT_DIRECTORY / "data" / "fashion-mnist_test.csv"
RESULTS_DIRECTORY = PROJECT_DIRECTORY / "results"

LABEL_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

LABEL_ORDER = [
    LABEL_NAMES[label]
    for label in sorted(LABEL_NAMES)
]

QUANTITATIVE_VARIABLES = [
    "mean_intensity",
    "pixel_std",
    "nonzero_pixels",
]


def load_data(
    train_path: Path,
    test_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and returns the training and testing datasets.
    """
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {train_path}"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Testing dataset not found at {test_path}"
        )

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    return train_data, test_data


def get_pixel_columns(data: pd.DataFrame) -> list[str]:
    """
    Returns the names of all pixel columns.
    """
    return [
        column
        for column in data.columns
        if column != "label"
    ]


def validate_dataset(data: pd.DataFrame, name: str) -> None:
    """
    Checks that a Fashion-MNIST dataset has the expected structure.
    """
    if "label" not in data.columns:
        raise ValueError(
            f"{name} dataset does not contain a label column."
        )

    pixel_columns = get_pixel_columns(data)

    if len(pixel_columns) != 784:
        raise ValueError(
            f"{name} dataset has {len(pixel_columns)} pixel columns. "
            "Expected 784."
        )

    valid_labels = set(LABEL_NAMES.keys())
    dataset_labels = set(data["label"].unique())

    if not dataset_labels.issubset(valid_labels):
        raise ValueError(
            f"{name} dataset contains an invalid label."
        )


def count_missing(data: pd.DataFrame) -> int:
    """
    Returns the total number of missing values in a dataset.
    """
    return int(data.isna().sum().sum())


def create_image_summary(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Creates image-level variables used for the EDA.
    """
    pixel_columns = get_pixel_columns(data)
    pixel_data = data[pixel_columns]

    result = pd.DataFrame()
    result["label"] = data["label"]
    result["category"] = data["label"].map(LABEL_NAMES)
    result["mean_intensity"] = pixel_data.mean(axis=1)
    result["pixel_std"] = pixel_data.std(axis=1)
    result["nonzero_pixels"] = pixel_data.gt(0).sum(axis=1)

    return result


def print_dataset_information(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame
) -> None:
    """
    Prints the dimensions, missing values, and pixel ranges.
    """
    print("DATASET DIMENSIONS")
    print("Training shape:", train_data.shape)
    print("Testing shape:", test_data.shape)

    print("\nMISSING VALUES")
    print(
        "Training missing values:",
        count_missing(train_data)
    )
    print(
        "Testing missing values:",
        count_missing(test_data)
    )

    train_pixels = train_data[get_pixel_columns(train_data)]
    test_pixels = test_data[get_pixel_columns(test_data)]

    print("\nPIXEL VALUE RANGES")
    print(
        "Training pixel minimum:",
        train_pixels.min().min()
    )
    print(
        "Training pixel maximum:",
        train_pixels.max().max()
    )
    print(
        "Testing pixel minimum:",
        test_pixels.min().min()
    )
    print(
        "Testing pixel maximum:",
        test_pixels.max().max()
    )


def print_variable_summaries(
    image_summary: pd.DataFrame
) -> None:
    """
    Prints categorical and quantitative variable summaries.
    """
    category_counts = (
        image_summary["category"]
        .value_counts()
        .reindex(LABEL_ORDER)
    )

    print("\nCATEGORY COUNTS")
    print(category_counts)

    seven_number_summary = (
        image_summary[QUANTITATIVE_VARIABLES]
        .describe()
        .loc[
            [
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
            ]
        ]
    )

    print("\nSEVEN-NUMBER SUMMARY")
    print(seven_number_summary)

    category_averages = (
        image_summary
        .groupby("category")[QUANTITATIVE_VARIABLES]
        .mean()
        .reindex(LABEL_ORDER)
    )

    print("\nAVERAGE VALUES BY CATEGORY")
    print(category_averages)


def plot_class_counts(
    image_summary: pd.DataFrame
) -> None:
    """
    Saves a bar plot showing the number of images in each category.
    """
    category_counts = (
        image_summary["category"]
        .value_counts()
        .reindex(LABEL_ORDER)
    )

    plt.figure(figsize=(11, 6))
    plt.bar(
        category_counts.index,
        category_counts.values
    )

    plt.title(
        "Number of Training Images by Clothing Category"
    )
    plt.xlabel("Clothing category")
    plt.ylabel("Number of images")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = RESULTS_DIRECTORY / "class_counts.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_intensity_by_category(
    image_summary: pd.DataFrame
) -> None:
    """
    Saves a box plot of mean pixel intensity by category.
    """
    intensity_values = []

    for category in LABEL_ORDER:
        category_data = image_summary[
            image_summary["category"] == category
        ]

        intensity_values.append(
            category_data["mean_intensity"]
        )

    plt.figure(figsize=(12, 6))
    plt.boxplot(
        intensity_values,
        tick_labels=LABEL_ORDER,
        showfliers=False
    )

    plt.title(
        "Mean Pixel Intensity by Clothing Category"
    )
    plt.xlabel("Clothing category")
    plt.ylabel("Mean pixel intensity")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = (
        RESULTS_DIRECTORY
        / "intensity_by_category.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_average_images(
    train_data: pd.DataFrame
) -> None:
    """
    Saves the average image for each clothing category.
    """
    pixel_columns = get_pixel_columns(train_data)
    figure, axes = plt.subplots(
        2,
        5,
        figsize=(12, 6)
    )

    for label, axis in enumerate(axes.flat):
        category_data = train_data[
            train_data["label"] == label
        ]

        average_pixels = (
            category_data[pixel_columns]
            .mean()
            .to_numpy()
        )

        average_image = average_pixels.reshape(28, 28)

        axis.imshow(
            average_image,
            cmap="gray"
        )
        axis.set_title(LABEL_NAMES[label])
        axis.axis("off")

    figure.suptitle(
        "Average Fashion-MNIST Image by Category"
    )

    plt.tight_layout(
        rect=(0, 0, 1, 0.95)
    )

    output_path = (
        RESULTS_DIRECTORY
        / "average_images.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    """
    Runs the complete Fashion-MNIST exploratory data analysis.
    """
    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    train_data, test_data = load_data(
        TRAIN_PATH,
        TEST_PATH
    )

    validate_dataset(
        train_data,
        "Training"
    )
    validate_dataset(
        test_data,
        "Testing"
    )

    print_dataset_information(
        train_data,
        test_data
    )

    image_summary = create_image_summary(
        train_data
    )

    print_variable_summaries(
        image_summary
    )

    plot_class_counts(
        image_summary
    )
    plot_intensity_by_category(
        image_summary
    )
    plot_average_images(
        train_data
    )

    print("\nEDA complete.")
    print(
        "Visualizations were saved in:",
        RESULTS_DIRECTORY
    )


if __name__ == "__main__":
    main()
