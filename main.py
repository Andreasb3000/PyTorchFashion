"""
Andreas Buneci
CSE 163
Summer 2026

Runs the Fashion-MNIST neural-network experiment, compares hidden
layer sizes, evaluates the best model, and creates result plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from modeling import (
    FashionClassifier,
    calculate_category_accuracy,
    create_confusion_matrix,
    create_data_loaders,
    evaluate_accuracy,
    get_predictions,
    load_model_data,
    train_model,
)


# Finds the folder that this Python file is located in.
# This makes the file paths work even if the program is started
# from somewhere else in the terminal.
PROJECT_DIRECTORY = Path(__file__).resolve().parent

# Paths to the training and testing CSV files.
TRAIN_PATH = (
    PROJECT_DIRECTORY
    / "data"
    / "fashion-mnist_train.csv"
)

TEST_PATH = (
    PROJECT_DIRECTORY
    / "data"
    / "fashion-mnist_test.csv"
)

# All plots created by this program will be stored here.
RESULTS_DIRECTORY = PROJECT_DIRECTORY / "results"

# These are the four hidden-layer sizes being compared for RQ2.
HIDDEN_SIZES = [32, 64, 128, 256]

# These settings stay the same for every model so that the only
# major difference between models is the hidden-layer size.
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
RANDOM_STATE = 42

# Maps each numerical Fashion-MNIST label to its clothing category.
# The position in this list matches the numerical label.
LABEL_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def plot_model_comparison(
    validation_accuracies: dict[int, float],
    training_times: dict[int, float]
) -> None:
    """
    Creates a plot comparing validation accuracy and training
    time for each hidden-layer size.
    """
    # Convert hidden sizes into strings so they can be used as
    # category labels on the bar graphs.
    labels = [
        str(hidden_size)
        for hidden_size in HIDDEN_SIZES
    ]

    # Convert accuracy from a decimal such as 0.88 into a percentage.
    accuracy_values = [
        validation_accuracies[hidden_size] * 100
        for hidden_size in HIDDEN_SIZES
    ]

    # Get the recorded training time for each hidden-layer size.
    time_values = [
        training_times[hidden_size]
        for hidden_size in HIDDEN_SIZES
    ]

    # Create two plots next to each other so accuracy and training
    # time can be compared in the same figure.
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4)
    )

    # First graph compares validation accuracy.
    axes[0].bar(
        labels,
        accuracy_values
    )

    axes[0].set_title(
        "Validation Accuracy by Hidden-Layer Size"
    )
    axes[0].set_xlabel("Hidden neurons")
    axes[0].set_ylabel("Validation accuracy (%)")

    # Second graph compares how long each model took to train.
    axes[1].bar(
        labels,
        time_values
    )

    axes[1].set_title(
        "Training Time by Hidden-Layer Size"
    )
    axes[1].set_xlabel("Hidden neurons")
    axes[1].set_ylabel("Training time (seconds)")

    # Adjust spacing so labels and titles do not overlap.
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIRECTORY
        / "model_comparison.png"
    )

    # Close the figure after saving so it does not stay in memory.
    plt.close()


def plot_loss_curves(
    loss_history: dict[int, list[float]]
) -> None:
    """
    Plots training loss across epochs for each model.
    """
    plt.figure(figsize=(8, 5))

    # Plot one loss curve for each hidden-layer size.
    # This shows how the model's error changes as training continues.
    for hidden_size in HIDDEN_SIZES:
        losses = loss_history[hidden_size]

        # Epoch numbers start at 1 rather than 0 for the graph.
        epochs = range(
            1,
            len(losses) + 1
        )

        plt.plot(
            epochs,
            losses,
            label=f"{hidden_size} neurons"
        )

    plt.title("Training Loss by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Average training loss")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIRECTORY
        / "loss_curves.png"
    )

    plt.close()


def plot_confusion_matrix(
    matrix: np.ndarray
) -> None:
    """
    Creates a visualization of the confusion matrix.
    """
    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    # Display the confusion matrix as an image where the color
    # represents the number of predictions in each cell.
    image = ax.imshow(matrix)

    # Add a color scale showing what the colors represent.
    fig.colorbar(
        image,
        ax=ax
    )

    ax.set_title(
        "Fashion-MNIST Test Confusion Matrix"
    )

    # Rows are actual classes while columns are predicted classes.
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")

    ax.set_xticks(
        range(len(LABEL_NAMES))
    )

    ax.set_yticks(
        range(len(LABEL_NAMES))
    )

    # Replace the numerical labels 0-9 with the actual clothing names.
    ax.set_xticklabels(
        LABEL_NAMES,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        LABEL_NAMES
    )

    # Write the exact number of predictions inside every matrix cell.
    # Diagonal cells represent correct predictions.
    for row in range(len(LABEL_NAMES)):
        for column in range(
            len(LABEL_NAMES)
        ):
            ax.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIRECTORY
        / "confusion_matrix.png"
    )

    plt.close()


def plot_category_accuracy(
    accuracies: list[float]
) -> None:
    """
    Creates a bar chart showing test accuracy for each category.
    """
    # Convert decimal accuracies into percentages for easier reading.
    percentages = [
        accuracy * 100
        for accuracy in accuracies
    ]

    plt.figure(figsize=(10, 5))

    # Each bar shows how accurately the model classified one category.
    plt.bar(
        LABEL_NAMES,
        percentages
    )

    plt.title(
        "Test Accuracy by Clothing Category"
    )

    plt.xlabel("Clothing category")
    plt.ylabel("Accuracy (%)")

    # Rotate category names so the longer labels are still readable.
    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIRECTORY
        / "per_category_accuracy.png"
    )

    plt.close()


def plot_misclassified_examples(
    model: FashionClassifier,
    test_loader,
    number_examples: int = 10
) -> None:
    """
    Displays examples that the final model classified incorrectly.
    """
    # Put the model into evaluation mode because no training
    # or weight updates happen in this function.
    model.eval()

    examples = []

    # Gradients are not needed because we are only making predictions.
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)

            # Choose the class with the highest output score.
            predictions = outputs.argmax(
                dim=1
            )

            # Go through the images in the current batch and save
            # examples where the predicted label is incorrect.
            for index in range(
                len(labels)
            ):
                if (
                    predictions[index]
                    != labels[index]
                ):
                    examples.append(
                        (
                            features[index],
                            labels[index].item(),
                            predictions[index].item()
                        )
                    )

                # Stop once enough incorrect examples have been collected.
                if (
                    len(examples)
                    == number_examples
                ):
                    break

            if (
                len(examples)
                == number_examples
            ):
                break

    # Display the 10 incorrect examples as a 2-by-5 grid.
    fig, axes = plt.subplots(
        2,
        5,
        figsize=(12, 5)
    )

    # Flatten turns the 2D axes array into one list that is easier
    # to loop through.
    axes = axes.flatten()

    for index, example in enumerate(
        examples
    ):
        image, true_label, predicted_label = (
            example
        )

        # The neural network stores each image as 784 values, so reshape
        # it back into its original 28-by-28 form for visualization.
        axes[index].imshow(
            image.reshape(28, 28),
            cmap="gray"
        )

        # Show both the correct label and what the model predicted.
        axes[index].set_title(
            f"True: {LABEL_NAMES[true_label]}\n"
            f"Pred: {LABEL_NAMES[predicted_label]}"
        )

        axes[index].axis("off")

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIRECTORY
        / "misclassified_examples.png"
    )

    plt.close()


def print_results(
    validation_accuracies: dict[int, float],
    training_times: dict[int, float],
    best_hidden_size: int,
    test_accuracy: float,
    category_accuracies: list[float]
) -> None:
    """
    Prints model comparison and final test results.
    """
    print()
    print("Model Comparison")
    print("----------------")

    # Print the validation accuracy and training time for each model.
    for hidden_size in HIDDEN_SIZES:
        print(
            f"{hidden_size} hidden neurons: "
            f"Validation Accuracy = "
            f"{validation_accuracies[hidden_size]:.4f}, "
            f"Training Time = "
            f"{training_times[hidden_size]:.2f} seconds"
        )

    print()

    # Report which model performed best on the validation data.
    print(
        f"Best hidden-layer size: "
        f"{best_hidden_size}"
    )

    # This is the final accuracy on the separate test dataset.
    print(
        f"Final test accuracy: "
        f"{test_accuracy:.4f}"
    )

    print()
    print("Per-Category Accuracy")
    print("---------------------")

    # Match each clothing category with its calculated test accuracy.
    for label, accuracy in zip(
        LABEL_NAMES,
        category_accuracies
    ):
        print(
            f"{label}: "
            f"{accuracy:.4f}"
        )


def main() -> None:
    """
    Runs the complete Fashion-MNIST experiment.
    """
    # Create the results folder if it does not already exist.
    RESULTS_DIRECTORY.mkdir(
        exist_ok=True
    )

    # Load both CSV files and convert them into PyTorch datasets.
    train_dataset, test_dataset = (
        load_model_data(
            str(TRAIN_PATH),
            str(TEST_PATH)
        )
    )

    # These dictionaries store the results from each hidden-layer size.
    # The hidden-layer size is used as the dictionary key.
    validation_accuracies = {}
    training_times = {}
    loss_history = {}
    trained_models = {}

    test_loader = None

    # Train one model for every hidden-layer size being compared.
    for hidden_size in HIDDEN_SIZES:
        print()
        print(
            f"Training model with "
            f"{hidden_size} hidden neurons"
        )

        # Reset the PyTorch random seed before each model so the
        # comparison between hidden sizes is reproducible.
        torch.manual_seed(
            RANDOM_STATE
        )

        # Create the same training and validation split for each model.
        (
            train_loader,
            validation_loader,
            test_loader
        ) = create_data_loaders(
            train_dataset,
            test_dataset,
            BATCH_SIZE,
            RANDOM_STATE
        )

        # Create a new neural network using the current hidden-layer size.
        model = FashionClassifier(
            hidden_size
        )

        # Train the model and collect its loss, validation accuracy,
        # and training time.
        (
            losses,
            model_validation_accuracies,
            training_time
        ) = train_model(
            model,
            train_loader,
            validation_loader,
            EPOCHS,
            LEARNING_RATE
        )

        # Use the validation accuracy from the final training epoch
        # when comparing the different models.
        final_validation_accuracy = (
            model_validation_accuracies[-1]
        )

        # Store the results so they can be compared after all models train.
        validation_accuracies[
            hidden_size
        ] = final_validation_accuracy

        training_times[
            hidden_size
        ] = training_time

        loss_history[
            hidden_size
        ] = losses

        trained_models[
            hidden_size
        ] = model

    # Find the hidden-layer size with the highest validation accuracy.
    # The test set is not used to choose the best model.
    best_hidden_size = max(
        validation_accuracies,
        key=validation_accuracies.get
    )

    # Retrieve the already-trained model that had the best
    # validation performance.
    best_model = trained_models[
        best_hidden_size
    ]

    # Evaluate the selected model on the official test dataset.
    # This gives the final accuracy used to answer RQ1.
    test_accuracy = evaluate_accuracy(
        best_model,
        test_loader
    )

    # Collect every true and predicted test label so we can analyze
    # which clothing categories the model confuses.
    (
        true_labels,
        predicted_labels
    ) = get_predictions(
        best_model,
        test_loader
    )

    # Build the confusion matrix using actual vs. predicted labels.
    confusion_matrix = (
        create_confusion_matrix(
            true_labels,
            predicted_labels
        )
    )

    # Use the confusion matrix to calculate accuracy for each
    # individual clothing category.
    category_accuracies = (
        calculate_category_accuracy(
            confusion_matrix
        )
    )

    # Create all visualizations needed for the final results section.
    plot_model_comparison(
        validation_accuracies,
        training_times
    )

    plot_loss_curves(
        loss_history
    )

    plot_confusion_matrix(
        confusion_matrix
    )

    plot_category_accuracy(
        category_accuracies
    )

    plot_misclassified_examples(
        best_model,
        test_loader
    )

    # Print the main numerical results after the experiment finishes.
    print_results(
        validation_accuracies,
        training_times,
        best_hidden_size,
        test_accuracy,
        category_accuracies
    )


if __name__ == "__main__":
    main()