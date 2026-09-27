
"""
Train a ResNet18 handwriting classifier using partial fine-tuning.

Training strategy:
- Start with ImageNet-pretrained ResNet18.
- Freeze:
    conv1
    bn1
    layer1
    layer2
    layer3
- Fine-tune:
    layer4
    fc

This is intended to reduce overfitting while allowing the deeper
ResNet features to adapt to handwriting images.
"""



from __future__ import annotations

import copy
import json
import logging
import os
import random
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import models

import sys
import os
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from models.handwriting.preprocessing import (
    build_transform,
    save_preprocessing_config,
)
from training.handwriting_dataset import (
    HandwritingTorchDataset,
    HandwritingSample,
    discover_handwriting_samples,
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# STUDENT / WRITER LEVEL SPLIT
# ============================================================

def student_level_split(
    samples: List[HandwritingSample],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = RANDOM_SEED,
) -> Tuple[
    List[HandwritingSample],
    List[HandwritingSample],
    List[HandwritingSample],
]:
    """
    Split data by student/writer when IDs are available.

    If no IDs are available, fall back to class-wise random splitting.

    NOTE:
    The current flat handwriting dataset does not appear to contain
    writer IDs in filenames, so the fallback split will normally be used.
    """

    rng = np.random.default_rng(seed)

    has_ids = all(
        sample.student_id is not None
        for sample in samples
    )

    if not has_ids:
        logger.warning(
            "No student/writer IDs could be extracted from filenames."
        )

        logger.warning(
            "Using a class-wise random image split. "
            "Samples from the same writer may therefore appear "
            "in both training and testing data."
        )

        by_class = defaultdict(list)

        for sample in samples:
            by_class[sample.class_name].append(sample)

        train = []
        val = []
        test = []

        for cls, items in by_class.items():

            indices = rng.permutation(len(items))

            shuffled_items = [
                items[i]
                for i in indices
            ]

            n = len(shuffled_items)

            n_train = int(n * train_frac)
            n_val = int(n * val_frac)

            train.extend(
                shuffled_items[:n_train]
            )

            val.extend(
                shuffled_items[
                    n_train:n_train + n_val
                ]
            )

            test.extend(
                shuffled_items[
                    n_train + n_val:
                ]
            )

        return train, val, test

    # --------------------------------------------------------
    # Writer-level split
    # --------------------------------------------------------

    by_student = defaultdict(list)

    for sample in samples:
        by_student[sample.student_id].append(sample)

    students = list(by_student.keys())

    rng.shuffle(students)

    n_students = len(students)

    n_train = int(n_students * train_frac)
    n_val = int(n_students * val_frac)

    train_students = students[:n_train]

    val_students = students[
        n_train:n_train + n_val
    ]

    test_students = students[
        n_train + n_val:
    ]

    train = []
    val = []
    test = []

    for student in train_students:
        train.extend(by_student[student])

    for student in val_students:
        val.extend(by_student[student])

    for student in test_students:
        test.extend(by_student[student])

    return train, val, test


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def log_class_distribution(
    samples: List[HandwritingSample],
    split_name: str,
) -> None:

    counts = defaultdict(int)

    for sample in samples:
        counts[sample.class_name] += 1

    logger.info(
        "%s class distribution: %s",
        split_name,
        dict(counts),
    )


# ============================================================
# MODEL
# ============================================================

def build_partial_finetune_model(
    num_classes: int,
) -> nn.Module:

    logger.info(
        "Loading ImageNet-pretrained ResNet18..."
    )

    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )

    # --------------------------------------------------------
    # Freeze entire model first
    # --------------------------------------------------------

    for parameter in model.parameters():
        parameter.requires_grad = False

    # --------------------------------------------------------
    # Replace final classifier
    # --------------------------------------------------------

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        num_classes,
    )

    # New FC layer must be trainable
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    # --------------------------------------------------------
    # Unfreeze only layer4
    # --------------------------------------------------------

    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    return model


# ============================================================
# PRINT TRAINABLE PARAMETERS
# ============================================================

def print_trainable_parameters(
    model: nn.Module,
) -> None:

    total_parameters = 0
    trainable_parameters = 0

    logger.info(
        "Trainable model parameters:"
    )

    for name, parameter in model.named_parameters():

        total_parameters += parameter.numel()

        if parameter.requires_grad:

            trainable_parameters += parameter.numel()

            logger.info(
                "  TRAINABLE: %s",
                name,
            )

        else:

            logger.info(
                "  FROZEN: %s",
                name,
            )

    logger.info(
        "Total parameters: %d",
        total_parameters,
    )

    logger.info(
        "Trainable parameters: %d",
        trainable_parameters,
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    samples: List[HandwritingSample],
    class_to_idx: dict,
) -> torch.Tensor:

    counts = np.zeros(
        len(class_to_idx),
        dtype=np.float32,
    )

    for sample in samples:

        class_index = class_to_idx[
            sample.class_name
        ]

        counts[class_index] += 1

    logger.info(
        "Training class counts: %s",
        counts.tolist(),
    )

    # --------------------------------------------------------
    # Balanced class weights
    #
    # weight = total_samples /
    #          (number_of_classes * class_count)
    # --------------------------------------------------------

    total = counts.sum()
    num_classes = len(counts)

    weights = total / (
        num_classes * counts
    )

    logger.info(
        "Class weights: %s",
        weights.tolist(),
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# ============================================================
# VALIDATION
# ============================================================

def evaluate_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predicted = torch.argmax(
                outputs,
                dim=1,
            )

            predictions.extend(
                predicted.cpu().numpy()
            )

            targets.extend(
                labels.cpu().numpy()
            )

    if len(targets) == 0:
        return 0.0

    return accuracy_score(
        targets,
        predictions,
    )


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    num_epochs: int,
    patience: int,
):
    best_val_accuracy = 0.0
    best_state = copy.deepcopy(
        model.state_dict()
    )

    epochs_without_improvement = 0

    for epoch in range(num_epochs):

        model.train()

        running_loss = 0.0
        total_samples = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # ------------------------------------------------
            # Clear gradients
            # ------------------------------------------------

            optimizer.zero_grad()

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            outputs = model(images)

            # ------------------------------------------------
            # Calculate loss
            # ------------------------------------------------

            loss = criterion(
                outputs,
                labels,
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            loss.backward()

            # ------------------------------------------------
            # Update weights
            # ------------------------------------------------

            optimizer.step()

            batch_size = images.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

        if total_samples > 0:

            train_loss = (
                running_loss /
                total_samples
            )

        else:
            train_loss = 0.0

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_accuracy = evaluate_validation(
            model,
            val_loader,
            device,
        )

        # ----------------------------------------------------
        # Learning-rate scheduler
        # ----------------------------------------------------

        scheduler.step(val_accuracy)

        logger.info(
            "Epoch %d/%d | "
            "Train Loss: %.4f | "
            "Validation Accuracy: %.4f",
            epoch + 1,
            num_epochs,
            train_loss,
            val_accuracy,
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

            logger.info(
                "New best validation accuracy: %.4f",
                best_val_accuracy,
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if epochs_without_improvement >= patience:

            logger.info(
                "Early stopping triggered after %d epochs "
                "without validation improvement.",
                patience,
            )

            break

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    model.load_state_dict(
        best_state
    )

    return model, best_val_accuracy


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

def evaluate_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_to_idx: dict,
) -> dict:

    model.eval()

    predictions = []
    targets = []
    probabilities = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            probs = torch.softmax(
                outputs,
                dim=1,
            )

            predicted = torch.argmax(
                probs,
                dim=1,
            )

            predictions.extend(
                predicted.cpu().numpy()
            )

            targets.extend(
                labels.numpy()
            )

            probabilities.extend(
                probs.cpu().numpy()
            )

    predictions = np.array(
        predictions
    )

    targets = np.array(
        targets
    )

    probabilities = np.array(
        probabilities
    )

    # --------------------------------------------------------
    # Basic metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    precision = precision_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        targets,
        predictions,
        labels=list(
            range(len(class_to_idx))
        ),
    )

    # --------------------------------------------------------
    # Sensitivity / specificity
    #
    # For binary classification:
    #
    # class 0 = non_dyslexic
    # class 1 = dyslexic
    # --------------------------------------------------------

    sensitivity = None
    specificity = None
    roc_auc = None

    if len(class_to_idx) == 2:

        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tp = cm[1][1]

        if (tp + fn) > 0:

            sensitivity = (
                tp /
                (tp + fn)
            )

        if (tn + fp) > 0:

            specificity = (
                tn /
                (tn + fp)
            )

        # Probability of positive class
        positive_probabilities = probabilities[:, 1]

        try:

            roc_auc = roc_auc_score(
                targets,
                positive_probabilities,
            )

        except ValueError:

            roc_auc = None

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": (
            float(roc_auc)
            if roc_auc is not None
            else None
        ),
        "sensitivity": (
            float(sensitivity)
            if sensitivity is not None
            else None
        ),
        "specificity": (
            float(specificity)
            if specificity is not None
            else None
        ),
        "confusion_matrix": cm.tolist(),
    }

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed()

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    project_root = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    data_dir = os.path.join(
        project_root,
        "data",
        "handwriting",
    )

    artifacts_dir = os.path.join(
        project_root,
        "artifacts",
        "handwriting",
    )

    os.makedirs(
        artifacts_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    image_size = 224

    batch_size = 16

    num_epochs = 30

    patience = 5

    layer4_learning_rate = 1e-5

    fc_learning_rate = 1e-4

    weight_decay = 1e-4

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    logger.info(
        "Using device: %s",
        device,
    )

    # --------------------------------------------------------
    # Discover dataset
    # --------------------------------------------------------

    logger.info(
        "Discovering handwriting dataset..."
    )

    samples = discover_handwriting_samples(
        data_dir
    )

    if len(samples) == 0:

        raise RuntimeError(
            "No handwriting samples were found."
        )

    logger.info(
        "Found %d handwriting samples.",
        len(samples),
    )

    # --------------------------------------------------------
    # Class mapping
    # --------------------------------------------------------

    class_names = sorted(
        set(
            sample.class_name
            for sample in samples
        )
    )

    class_to_idx = {
        class_name: index
        for index, class_name
        in enumerate(class_names)
    }

    logger.info(
        "Class mapping: %s",
        class_to_idx,
    )

    # --------------------------------------------------------
    # Split dataset
    # --------------------------------------------------------

    train_samples, val_samples, test_samples = (
        student_level_split(
            samples,
            train_frac=0.70,
            val_frac=0.15,
            seed=RANDOM_SEED,
        )
    )

    logger.info(
        "Train samples: %d",
        len(train_samples),
    )

    logger.info(
        "Validation samples: %d",
        len(val_samples),
    )

    logger.info(
        "Test samples: %d",
        len(test_samples),
    )

    log_class_distribution(
        train_samples,
        "Training",
    )

    log_class_distribution(
        val_samples,
        "Validation",
    )

    log_class_distribution(
        test_samples,
        "Test",
    )

    # --------------------------------------------------------
    # Transforms
    # --------------------------------------------------------

    train_transform = build_transform(
        image_size=image_size,
        augment=True,
    )

    eval_transform = build_transform(
        image_size=image_size,
        augment=False,
    )

    # --------------------------------------------------------
    # Dataset objects
    # --------------------------------------------------------

    train_dataset = HandwritingTorchDataset(
        train_samples,
        class_to_idx,
        transform=train_transform,
    )

    val_dataset = HandwritingTorchDataset(
        val_samples,
        class_to_idx,
        transform=eval_transform,
    )

    test_dataset = HandwritingTorchDataset(
        test_samples,
        class_to_idx,
        transform=eval_transform,
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Build partial fine-tuning model
    # --------------------------------------------------------

    model = build_partial_finetune_model(
        num_classes=len(class_names)
    )

    model = model.to(device)

    print_trainable_parameters(
        model
    )

    # --------------------------------------------------------
    # Class-weighted loss
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        train_samples,
        class_to_idx,
    )

    class_weights = class_weights.to(
        device
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # Optimizer
    #
    # layer4 gets a smaller learning rate
    # because it contains pretrained features.
    #
    # fc gets a larger learning rate because
    # it is newly initialized.
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.layer4.parameters(),
                "lr": layer4_learning_rate,
            },
            {
                "params": model.fc.parameters(),
                "lr": fc_learning_rate,
            },
        ],
        weight_decay=weight_decay,
    )

    # --------------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    logger.info(
        "Starting partial fine-tuning..."
    )

    model, best_val_accuracy = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        patience=patience,
    )

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    logger.info(
        "Evaluating best model on test set..."
    )

    metrics = evaluate_test(
        model,
        test_loader,
        device,
        class_to_idx,
    )

    # --------------------------------------------------------
    # Add experiment information
    # --------------------------------------------------------

    metrics["num_train_samples"] = len(
        train_samples
    )

    metrics["num_val_samples"] = len(
        val_samples
    )

    metrics["num_test_samples"] = len(
        test_samples
    )

    metrics["num_classes"] = len(
        class_names
    )

    metrics["best_val_accuracy"] = float(
        best_val_accuracy
    )

    metrics["training_strategy"] = (
        "partial_fine_tuning"
    )

    metrics["frozen_layers"] = [
        "conv1",
        "bn1",
        "layer1",
        "layer2",
        "layer3",
    ]

    metrics["trainable_layers"] = [
        "layer4",
        "fc",
    ]

    metrics["layer4_learning_rate"] = (
        layer4_learning_rate
    )

    metrics["fc_learning_rate"] = (
        fc_learning_rate
    )

    metrics["weight_decay"] = (
        weight_decay
    )

    metrics["early_stopping_patience"] = (
        patience
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    logger.info(
        "================ FINAL RESULTS ================"
    )

    logger.info(
        "Accuracy: %.4f",
        metrics["accuracy"],
    )

    logger.info(
        "Precision: %.4f",
        metrics["precision"],
    )

    logger.info(
        "Recall: %.4f",
        metrics["recall"],
    )

    logger.info(
        "F1 Score: %.4f",
        metrics["f1_score"],
    )

    logger.info(
        "ROC-AUC: %s",
        metrics["roc_auc"],
    )

    logger.info(
        "Sensitivity: %s",
        metrics["sensitivity"],
    )

    logger.info(
        "Specificity: %s",
        metrics["specificity"],
    )

    logger.info(
        "Confusion Matrix: %s",
        metrics["confusion_matrix"],
    )

    logger.info(
        "==============================================="
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = os.path.join(
        artifacts_dir,
        "resnet18.pth",
    )

    torch.save(
        model.state_dict(),
        model_path,
    )

    logger.info(
        "Saved model: %s",
        model_path,
    )

    # --------------------------------------------------------
    # Save class mapping
    # --------------------------------------------------------

    class_mapping_path = os.path.join(
        artifacts_dir,
        "class_mapping.json",
    )

    with open(
        class_mapping_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            class_to_idx,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save preprocessing configuration
    # --------------------------------------------------------

    preprocessing_path = os.path.join(
        artifacts_dir,
        "preprocessing.json",
    )

    save_preprocessing_config(
        preprocessing_path,
        image_size=image_size,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_path = os.path.join(
        artifacts_dir,
        "metrics.json",
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    logger.info(
        "Saved metrics: %s",
        metrics_path,
    )

    logger.info(
        "Handwriting training completed."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
