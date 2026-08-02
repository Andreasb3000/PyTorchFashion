"""
Loads the Fashion-MNIST datasets.
"""

import pandas as pd


TRAIN_PATH = "data/fashion-mnist_train.csv"
TEST_PATH = "data/fashion-mnist_test.csv"


def main() -> None:
    """
    Loads the datasets and prints their dimensions.
    """
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    print("Training shape:", train_data.shape)
    print("Testing shape:", test_data.shape)


if __name__ == "__main__":
    main()