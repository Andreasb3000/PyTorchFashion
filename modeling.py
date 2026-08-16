"""
Andreas Buneci
CSE 163
Summer 2026

Loads Fashion-MNIST data, creates PyTorch datasets and data loaders,
trains a neural network, and evaluates classification performance.
"""

import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


INPUT_SIZE = 784
NUM_CLASSES = 10
VALIDATION_SIZE = 10000


def load_model_data(
    train_path: str,
    test_path: str
) -> tuple[TensorDataset, TensorDataset]:
    """
    Loads the CSV files and converts them into PyTorch TensorDatasets.
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Convert the data into PyTorch tensors and normalize pixel values
    # from 0-255 to 0-1.
    train_features = torch.tensor(
        train_data.drop(columns=["label"]).values,
        dtype=torch.float32
    ) / 255.0

    train_labels = torch.tensor(
        train_data["label"].values,
        dtype=torch.long
    )

    test_features = torch.tensor(
        test_data.drop(columns=["label"]).values,
        dtype=torch.float32
    ) / 255.0

    test_labels = torch.tensor(
        test_data["label"].values,
        dtype=torch.long
    )

    # Connect each image tensor to its corresponding label.
    train_dataset = TensorDataset(
        train_features,
        train_labels
    )

    test_dataset = TensorDataset(
        test_features,
        test_labels
    )

    return train_dataset, test_dataset


def create_data_loaders(
    train_dataset: TensorDataset,
    test_dataset: TensorDataset,
    batch_size: int,
    random_state: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Splits the training dataset into training and validation sets
    and creates DataLoaders.
    """
    train_size = len(train_dataset) - VALIDATION_SIZE

    # Control which samples go into training and validation so the split
    # is reproducible across runs.
    generator = torch.Generator().manual_seed(random_state)

    training_data, validation_data = random_split(
        train_dataset,
        [train_size, VALIDATION_SIZE],
        generator=generator
    )

    shuffle_generator = torch.Generator().manual_seed(random_state)

    # Train in batches rather than processing the entire dataset at once.
    # The training loader shuffles examples, while validation and testing
    # preserve their existing order.
    train_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_generator
    )

    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, validation_loader, test_loader


class FashionClassifier(nn.Module):
    """
    Neural network for classifying Fashion-MNIST images.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()

        # The input layer has 784 values for a 28-by-28 image. The hidden
        # layer size is varied across models, and the output has 10 classes.
        self._network = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, NUM_CLASSES)
        )

        # The first linear layer maps the 784 input pixels to the hidden
        # layer. ReLU introduces non-linearity by replacing negative values
        # with zero. The second linear layer maps the hidden layer to the
        # 10 output class scores.
        #
        # A larger hidden layer gives the network more parameters and more
        # capacity to learn complicated patterns.

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Passes features through the neural network.
        """
        return self._network(features)


def evaluate_accuracy(
    model: FashionClassifier,
    data_loader: DataLoader
) -> float:
    """
    Returns the classification accuracy of a model.
    """
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in data_loader:
            outputs = model(features)

            # Choose the class with the highest output score.
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


def train_model(
    model: FashionClassifier,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
    learning_rate: float
) -> tuple[list[float], list[float], float]:
    """
    Trains a model and returns training losses, validation accuracies,
    and training time.
    """
    # Measure how wrong the model's predictions are. Lower loss is better.
    loss_function = nn.CrossEntropyLoss()

    # Adam updates model weights and biases to reduce loss. The learning
    # rate controls how aggressively those parameters are updated.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    losses = []
    validation_accuracies = []

    # Record when training starts.
    start_time = time.perf_counter()

    # Each epoch is one complete pass through the training dataset.
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # This loop performs the model's learning one batch at a time.
        for features, labels in train_loader:
            # Clear gradients left over from the previous batch.
            optimizer.zero_grad()

            # Make predictions for the current batch.
            outputs = model(features)

            # Measure the error in the current predictions.
            loss = loss_function(outputs, labels)

            # Backpropagation calculates how each parameter contributed to
            # the loss and the direction it should move to reduce that loss.
            loss.backward()

            # Update the parameters using the calculated gradients.
            optimizer.step()

            # Convert the batch's average loss back into total batch loss so
            # the epoch-wide average can be calculated correctly.
            total_loss += loss.item() * labels.size(0)

        average_loss = total_loss / len(train_loader.dataset)

        # Validation accuracy measures performance on data not used to update
        # the model's weights.
        validation_accuracy = evaluate_accuracy(
            model,
            validation_loader
        )

        losses.append(average_loss)
        validation_accuracies.append(validation_accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Loss: {average_loss:.4f} - "
            f"Validation Accuracy: {validation_accuracy:.4f}"
        )

    # Stop the timer after all epochs finish.
    training_time = time.perf_counter() - start_time

    return losses, validation_accuracies, training_time


def get_predictions(
    model: FashionClassifier,
    data_loader: DataLoader
) -> tuple[list[int], list[int]]:
    """
    Returns the true labels and predicted labels for a dataset.
    """
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for features, labels in data_loader:
            outputs = model(features)

            # Return the index of the largest value as the predicted label.
            predictions = outputs.argmax(dim=1)

            true_labels.extend(labels.tolist())
            predicted_labels.extend(predictions.tolist())

    return true_labels, predicted_labels


def create_confusion_matrix(
    true_labels: list[int],
    predicted_labels: list[int]
) -> np.ndarray:
    """
    Creates a confusion matrix from true and predicted labels.
    """
    # Rows represent actual classes and columns represent predicted classes.
    # Diagonal values count correct predictions, while off-diagonal values
    # show which classes the model confuses with one another.
    matrix = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=int
    )

    for true_label, predicted_label in zip(
        true_labels,
        predicted_labels
    ):
        matrix[true_label, predicted_label] += 1

    return matrix


def calculate_category_accuracy(
    confusion_matrix: np.ndarray
) -> list[float]:
    """
    Calculates classification accuracy for each clothing category.
    """
    accuracies = []

    for category in range(NUM_CLASSES):
        correct = confusion_matrix[
            category,
            category
        ]

        total = confusion_matrix[category].sum()
        accuracies.append(correct / total)

    return accuracies