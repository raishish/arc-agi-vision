import torch
from torch.utils.data import random_split, Dataset, DataLoader
from torch.nn import functional as F
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from typing import List
import os
import pandas as pd
from losses import get_loss
import ast


class ARCDataset(Dataset):
    """ARC Dataset"""
    def __init__(self, dataframe, transform=None):
        """Initialize the ARCDataset class.

        Args:
            dataframe (pandas.DataFrame): dataframe containing the input and output grids
            transform (torch.nn.Module, optional): transform to apply to the input and output grids.
        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        input_grid = torch.tensor(
            eval(self.dataframe.iloc[idx]['input']), dtype=torch.float32)
        output_grid = torch.tensor(
            eval(self.dataframe.iloc[idx]['output']), dtype=torch.float32)
        metadata = {
            'sample_id': self.dataframe.iloc[idx]['sample_id'],
            'input_dims': (input_grid.shape[0], input_grid.shape[1]),
            'output_dims': (output_grid.shape[0], output_grid.shape[1])
        }

        if self.transform:
            input_grid = self.transform(input_grid)
            output_grid = self.transform(output_grid)

        return metadata, input_grid.unsqueeze(0), output_grid.unsqueeze(0)


class FineTuneDataset(Dataset):
    """Dataset used for fine-tuning"""
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pandas.DataFrame): dataframe containing the input and output grids
            transform (torch.nn.Module, optional): transform to apply to the input and output grids.
        """
        self.dataframe = dataframe
        self.transform = transform
        self.dataframe['input'] = self.dataframe['input'].apply(ast.literal_eval)
        self.dataframe['output'] = self.dataframe['output'].apply(ast.literal_eval)
        self.groups = list(dataframe.groupby('sample_id'))
        self.groups = [group for group in self.groups if len(group[1]) > 1]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        sample_id, group = self.groups[idx]
        test_row = group.iloc[0]    # the first pair is the test pair
        train_rows = group.iloc[1:]

        metadata = {
            'sample_id': sample_id,
            'test': {
                'input_dims': (int(test_row['input_grid_width']), int(test_row['input_grid_height'])),
                'output_dims': (int(test_row['output_grid_width']), int(test_row['output_grid_height']))
            },
            'train': [
                {
                    'input_dims': (int(row['input_grid_width']), int(row['input_grid_height'])),
                    'output_dims': (int(row['output_grid_width']), int(row['output_grid_height']))
                }
                for _, row in train_rows.iterrows()
            ]
        }

        # Get the test pair
        test_input, test_output = test_row[['input', 'output']]
        test_input = torch.tensor(test_input, dtype=torch.float32)
        test_output = torch.tensor(test_output, dtype=torch.float32)

        # Get the train pairs
        train_inputs = train_rows['input'].tolist()
        train_outputs = train_rows['output'].tolist()
        train_inputs = [torch.tensor(train_input, dtype=torch.float32) for train_input in train_inputs]
        train_outputs = [torch.tensor(train_output, dtype=torch.float32) for train_output in train_outputs]

        if self.transform:
            train_inputs = [self.transform(train_input).unsqueeze(0) for train_input in train_inputs]
            train_inputs = torch.cat(train_inputs, dim=0)

            train_outputs = [self.transform(train_output).unsqueeze(0) for train_output in train_outputs]
            train_outputs = torch.cat(train_outputs, dim=0)

        return metadata, (test_input, test_output), (train_inputs, train_outputs)


class PadTransform:
    """Pads the input and output grids to the target size"""
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, grid):
        # Calculate padding
        pad_height = self.target_size[0] - grid.shape[0]
        pad_width = self.target_size[1] - grid.shape[1]

        # Calculate padding for both sides
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Pad the grid equally on both sides
        padded_grid = F.pad(
            grid, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return padded_grid


class UnpadTransform:
    """Unpads the input and output grids to the original size"""
    def __call__(self, grid, original_size):
        h, w = original_size
        pad_h = (grid.shape[0] - h) // 2
        pad_w = (grid.shape[1] - w) // 2

        unpadded_grid = grid[pad_h:pad_h + h, pad_w:pad_w + w]
        return unpadded_grid


def get_class_weights(batch_targets: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """
    Get class weights for a given batch of targets.

    Args:
        batch_targets (torch.Tensor): batch of targets
        num_classes (int, optional): number of classes. Defaults to 10.

    Returns:
        torch.Tensor: class weights
    """
    unique_values, counts = torch.unique(batch_targets, return_counts=True)
    freqs = counts / counts.sum()
    class_weights = torch.ones(num_classes).to(batch_targets.device)
    class_weights[unique_values.long()] = 1.0 / freqs

    return class_weights


def get_fine_tune_dataloader(dataframe, transform=PadTransform((30, 30)), shuffle=False):
    """
    Creates a dataloader for fine-tuning.

    Args:
        dataframe (pandas.DataFrame): dataframe containing the input and output grids
        transform (torch.nn.Module, optional): transform to apply to the input and output grids
        shuffle (bool, optional): whether to shuffle the data

    Returns:
        torch.utils.data.DataLoader: dataloader
    """
    dataset = FineTuneDataset(dataframe, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
    return dataloader


def get_dataloaders(
    dataframe,
    val_split=0.2,
    batch_size=32,
    transform=PadTransform((30, 30)),
    shuffle=True
):
    """
    Creates and returns dataloaders for training and validation/testing.

    Args:
        dataframe (pandas.DataFrame): dataframe containing the input and output grids
        val_split (float, optional): fraction of the dataset to use for validation/testing. Defaults to 0.2.
        batch_size (int, optional): Batch size
        transform (_type_, optional): transform to apply to the input and output grids
        shuffle (bool, optional): shuffle the dataset

    Returns:
        List[torch.utils.data.DataLoader]: list of dataloaders for training and validation/testing
    """
    dataset = ARCDataset(dataframe, transform=transform)

    if val_split is None or val_split == 0.:  # No validation set / Test set
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        print("Dataset size: ", len(dataset))
        return dataloader

    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    return train_dataloader, val_dataloader


def process_one_batch(
    model,
    data: tuple,
    optimizer: torch.optim.Optimizer,
    loss_criteria: List[str] = ['ce'],
    loss_weights: List[float] = [1.],
    acc_criterion=None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    mode: str = "eval",
):
    """
    Processes one batch of data.

    Args:
        data: tuple of (input, target) tensors
        optimizer: optimizer
        loss_criterion: loss criterion
        acc_criterion (optional): accuracy criterion
        device (torch.device): device
        mode (str): mode to use (train, eval, or test)

    Returns:
        tuple: (outputs, metrics)
    """
    # Move data to the device
    inputs, targets = data[0].to(device), data[1].to(device).squeeze()
    class_weights = get_class_weights(targets)

    # Forward pass
    logits, outputs = model.forward(inputs, return_logits=True)

    # Calculate loss
    total_loss = 0
    losses = {}
    for loss_criterion, loss_weight in zip(loss_criteria, loss_weights):
        loss_function = get_loss(loss_criterion, class_weights=class_weights)
        loss = loss_function(logits, targets.long())
        losses[f"{loss_criterion}_loss"] = loss
        total_loss += loss * loss_weight

    # Backpropagate
    if mode == "train":
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Calculate accuracy
    if acc_criterion:
        batch_acc = acc_criterion(outputs, targets)
    else:
        batch_acc = None

    metrics = get_accuracy_metrics(outputs, targets)
    metrics["loss"] = total_loss.item()
    metrics.update(losses)
    metrics["accuracy"] = batch_acc.item()

    return outputs, metrics


def train_model(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler = None,
    loss_criteria: List[str] = ['ce'],
    loss_weights: List[float] = [1.],
    acc_criterion: torch.nn.Module = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_model: bool = True,
    save_folder: str = "models",
    wandb_tracking: bool = True,
    test: bool = True,
    test_dataloader: torch.utils.data.DataLoader = None,
    **params
):
    """
    Trains a model on the given dataloaders.

    Args:
        model (torch.nn.Module): model to train
        train_dataloader (torch.utils.data.DataLoader): dataloader for training
        val_dataloader (torch.utils.data.DataLoader): dataloader for validation/testing
        epochs (int): number of epochs to train the model
        optimizer (torch.optim.Optimizer): optimizer
        scheduler (torch.optim.lr_scheduler, optional): learning rate scheduler
        loss_criteria (List[str]): List of loss criteria to use
        loss_weights (List[float]): List of loss weights
        acc_criterion (torch.nn.Module, optional): accuracy criterion
        device (torch.device, optional): device to use for training
        save_folder (str, optional): folder to save the model
        wandb_tracking (bool, optional): whether to track metrics with wandb
        test (bool, optional): whether to test the model
        params (dict, optional): additional parameters
    """

    if wandb_tracking:
        # Initialize wandb
        wandb.init(
            entity=params.get("wandb_entity", "arc-agi-vision"),
            project=params.get("wandb_project", "arc-agi-vision"),
            config=params
        )

    learning_rate = optimizer.param_groups[0]['lr']  # Save the learning rate
    model.to(device)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_metrics = {
            "loss": 0.0,
            "ce_loss": 0.0,
            "kl_loss": 0.0,
            "grid_loss": 0.0,
            "focal_loss": 0.0,
            "dice_loss": 0.0,
            "tversky_loss": 0.0,
            "accuracy": 0.0,
            "total_pixels": 0,
            "correct_pixels": 0,
            "total_grids": 0,
            "correct_grids": 0,
            "total_background_pixels": 0,
            "correct_background_pixels": 0,
            "total_foreground_pixels": 0,
            "correct_foreground_pixels": 0,
            "num_batches": 0
        }

        with tqdm(total=len(train_dataloader), desc=f"Train Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for _, inputs, targets in train_dataloader:
                _, batch_metrics = process_one_batch(
                    model,
                    (inputs, targets),
                    optimizer,
                    loss_criteria=loss_criteria,
                    loss_weights=loss_weights,
                    acc_criterion=acc_criterion,
                    device=device,
                    mode="train"
                )

                aggregate_metrics(train_metrics, batch_metrics)

                # Log loss and accuracy metrics
                pbar.set_postfix({
                    key: f"{value:.2f}"
                    for key, value in batch_metrics.items()
                    if "loss" in key or "accuracy" in key
                })
                pbar.update(1)

        # Validation phase
        model.eval()
        val_metrics = {
            "loss": 0.0,
            "ce_loss": 0.0,
            "kl_loss": 0.0,
            "grid_loss": 0.0,
            "focal_loss": 0.0,
            "dice_loss": 0.0,
            "tversky_loss": 0.0,
            "accuracy": 0.0,
            "total_pixels": 0,
            "correct_pixels": 0,
            "total_grids": 0,
            "correct_grids": 0,
            "total_background_pixels": 0,
            "correct_background_pixels": 0,
            "total_foreground_pixels": 0,
            "correct_foreground_pixels": 0,
            "num_batches": 0
        }

        with tqdm(total=len(val_dataloader), desc=f"Validation Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for _, inputs, targets in val_dataloader:
                _, batch_metrics = process_one_batch(
                    model,
                    (inputs, targets),
                    optimizer,
                    loss_criteria=loss_criteria,
                    loss_weights=loss_weights,
                    acc_criterion=acc_criterion,
                    device=device,
                    mode="eval"
                )

                aggregate_metrics(val_metrics, batch_metrics)

                # Log loss and accuracy metrics
                pbar.set_postfix({
                    key: f"{value:.2f}"
                    for key, value in batch_metrics.items()
                    if "loss" in key or "accuracy" in key
                })
                pbar.update(1)

        if wandb_tracking:
            log_metrics_to_wandb(train_metrics, type="train", step=epoch)
            log_metrics_to_wandb(val_metrics, type="val", step=epoch)

        # Step the scheduler if provided
        if scheduler:
            scheduler.step(val_metrics["loss"])
            wandb.log({"lr": optimizer.param_groups[0]['lr']}, step=epoch)

    # Save the model
    os.makedirs(save_folder, exist_ok=True)
    model_name = f"{params['architecture']}_{epochs}epochs_lr{learning_rate}.pt"
    model_path = os.path.join(save_folder, model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    if test:
        eval_model(
            model,
            test_dataloader,
            loss_criteria=loss_criteria,
            loss_weights=loss_weights,
            acc_criterion=acc_criterion,
            device=device,
            save_results=True,
            model_name=model_name,
            results_dir="results",
            wandb_tracking=wandb_tracking
        )

    if wandb_tracking:
        wandb.save(model_path)
        wandb.finish()


def eval_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_criteria: List[str] = ['ce'],
    loss_weights: List[float] = [1.],
    acc_criterion: torch.nn.Module = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_results: bool = True,
    model_name: str = "vision_model.pt",
    results_dir: str = "results",
    wandb_tracking: bool = False
) -> dict:
    """
    Evaluates a model on a given dataloader.
    Saves results to a CSV file if save_results is True.

    Args:
        model (torch.nn.Module): model to evaluate
        dataloader (torch.utils.data.DataLoader): dataloader
        device (torch.device): device to use for evaluation

    Returns:
        dict: evaluation metrics
    """
    eval_metrics = {
        "loss": 0.0,
        "ce_loss": 0.0,
        "kl_loss": 0.0,
        "grid_loss": 0.0,
        "focal_loss": 0.0,
        "dice_loss": 0.0,
        "tversky_loss": 0.0,
        "accuracy": 0.0,
        "total_pixels": 0,
        "correct_pixels": 0,
        "total_grids": 0,
        "correct_grids": 0,
        "total_background_pixels": 0,
        "correct_background_pixels": 0,
        "total_foreground_pixels": 0,
        "correct_foreground_pixels": 0,
        "num_batches": 0
    }

    results = {
        "sample_id": [],
        "input": [],
        "target": [],
        "predicted": [],
        "padded_input": [],
        "padded_target": [],
        "padded_predicted": []
    }

    model.eval()
    unpad_transform = UnpadTransform()
    with tqdm(total=len(dataloader), desc="Evaluation", unit="batch") as pbar:
        for metadata, inputs, targets in dataloader:
            outputs, batch_metrics = process_one_batch(
                model,
                (inputs, targets),
                None,
                loss_criteria=loss_criteria,
                loss_weights=loss_weights,
                acc_criterion=acc_criterion,
                device=device,
                mode="eval"
            )
            aggregate_metrics(eval_metrics, batch_metrics)

            results["sample_id"] += metadata["sample_id"]

            inputs = inputs.squeeze().int()
            targets = targets.squeeze().int()
            outputs = outputs.argmax(1).squeeze().int()

            results["padded_input"] += inputs.cpu().numpy().tolist()
            results["padded_target"] += targets.cpu().numpy().tolist()
            results["padded_predicted"] += outputs.cpu().numpy().tolist()

            # unpad the input, output, and target grids
            unpadded_inputs = [unpad_transform(grid, input_dims) for grid, input_dims in zip(
                inputs, zip(*metadata["input_dims"]))]
            unpadded_targets = [unpad_transform(grid, output_dims) for grid, output_dims in zip(
                targets, zip(*metadata["output_dims"]))]
            unpadded_outputs = [unpad_transform(grid, output_dims) for grid, output_dims in zip(
                outputs, zip(*metadata["output_dims"]))]

            results["input"] += [unpadded_input.cpu().numpy().tolist() for unpadded_input in unpadded_inputs]
            results["target"] += [unpadded_target.cpu().numpy().tolist() for unpadded_target in unpadded_targets]
            results["predicted"] += [unpadded_output.cpu().numpy().tolist() for unpadded_output in unpadded_outputs]

            # Log loss and accuracy metrics
            pbar.set_postfix({
                key: f"{value:.2f}"
                for key, value in batch_metrics.items()
                if "loss" in key or "accuracy" in key
            })
            pbar.update(1)

    # Print metrics
    filtered_metrics = [key for key in eval_metrics.keys() if "loss" in key or "accuracy" in key]
    print(eval_metrics)
    print("Evaluation Metrics:")
    print("-" * 50)
    for metric in filtered_metrics:
        print(f"{metric}: {eval_metrics[metric]:.2f}")

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{model_name.split('.')[0]}_eval_results.csv")
    test_df = pd.DataFrame(results)
    test_df['is_correct'] = test_df.apply(lambda row: row['predicted'] == row['target'], axis=1)
    test_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    if wandb_tracking:
        log_metrics_to_wandb(eval_metrics, type="eval", step=None)
        wandb.save(results_file)


def plot_batch(models: List[torch.nn.Module], sample_ids: list, inputs: torch.Tensor, targets: torch.Tensor):
    """
    Plots the predicted grids for a batch of inputs and targets.

    Args:
        models (List[torch.nn.Module]): list of models
        inputs (torch.Tensor): batch of inputs
        targets (torch.Tensor): batch of targets
    """
    predicted_grids = []
    model_names = [model.__class__.__name__ for model in models]

    for model, model_name in zip(models, model_names):
        model.eval()
        outputs = model(inputs)
        predicted = outputs.argmax(1)

        metrics = get_accuracy_metrics(outputs, targets)

        print(f"\n{model_name}")
        print("-" * 50)
        print(f"Pixel accuracy: {100. * metrics['pixel_accuracy']:.2f} %")
        print(f"Grid accuracy: {100. * metrics['grid_accuracy']:.2f} %")
        print(f"Background accuracy: {100. * metrics['background_accuracy']:.2f} %")
        print(f"Foreground accuracy: {100. * metrics['foreground_accuracy']:.2f} %")
        print("=" * 50)

        predicted_grids.append(predicted)

    predicted_grids = torch.stack(predicted_grids)
    # Move batch dim to the first dim (batch_sz, num_models, grid_sz, grid_sz)
    predicted_grids = predicted_grids.transpose(0, 1)

    fig, axs = plt.subplots(len(inputs), 2 + len(models), figsize=(2 * (2 + len(models)), 2 * len(inputs)))

    for i, (sample_id, input_grid, target_grid, predicted_grid) in enumerate(zip(sample_ids, inputs, targets, predicted_grids)):
        axs[i][0].imshow(input_grid.squeeze().numpy())
        axs[i][0].set_title('Input - ' + sample_id)
        axs[i][0].axis('off')

        axs[i][1].imshow(target_grid.squeeze().numpy())
        axs[i][1].set_title('Target Grid')
        axs[i][1].axis('off')

        for j in range(len(models)):
            axs[i][2 + j].imshow(predicted_grid[j].squeeze().detach().numpy())
            axs[i][2 + j].set_title(model_names[j])
            axs[i][2 + j].axis('off')

    plt.tight_layout()
    plt.show()


def get_accuracy_metrics(outputs: torch.Tensor, targets: torch.Tensor, background_class_idx: int = 0) -> dict:
    """Get accuracy metrics for a batch of outputs and targets.

    Args:
        outputs (torch.Tensor): batch of outputs
        targets (torch.Tensor): batch of targets
        background_class_idx (int, optional): index of the background class. Defaults to 0.

    Returns:
        dict: accuracy metrics
    """
    # Calculate pixel accuracy
    predicted = outputs.argmax(1)  # Get the predicted class for each pixel
    total_pixels = targets.numel()
    correct_pixels = predicted.eq(targets).sum().item()
    pixel_accuracy = correct_pixels / total_pixels

    # Calculate grid accuracy
    total_grids = targets.size(0)
    equality = predicted == targets
    correct_grids = torch.all(equality, dim=(1, 2)).sum().item()
    grid_accuracy = correct_grids / total_grids

    # Calculate background and foreground accuracy
    background_mask = targets == background_class_idx
    foreground_mask = targets != background_class_idx

    total_background_pixels = background_mask.sum().item()
    total_foreground_pixels = foreground_mask.sum().item()

    correct_background_pixels = (predicted[background_mask] == background_class_idx).sum().item()
    correct_foreground_pixels = (predicted[foreground_mask] == targets[foreground_mask]).sum().item()

    background_accuracy = correct_background_pixels / total_background_pixels if total_background_pixels > 0 else 0.
    foreground_accuracy = correct_foreground_pixels / total_foreground_pixels if total_foreground_pixels > 0 else 0.

    return {
        "total_pixels": total_pixels,
        "correct_pixels": correct_pixels,
        "pixel_accuracy": pixel_accuracy,
        "total_grids": total_grids,
        "correct_grids": correct_grids,
        "grid_accuracy": grid_accuracy,
        "total_background_pixels": total_background_pixels,
        "correct_background_pixels": correct_background_pixels,
        "background_accuracy": background_accuracy,
        "total_foreground_pixels": total_foreground_pixels,
        "correct_foreground_pixels": correct_foreground_pixels,
        "foreground_accuracy": foreground_accuracy
    }


def aggregate_metrics(metrics: dict, batch_metrics: dict):
    """Aggregates batch metrics into the metrics dictionary"""
    loss_keys = [key for key in batch_metrics.keys() if "loss" in key]
    for loss_key in loss_keys:
        metrics[loss_key] = (metrics[loss_key] * metrics["num_batches"] + batch_metrics[loss_key]) / \
                             (metrics["num_batches"] + 1)

    metrics["total_pixels"] += batch_metrics["total_pixels"]
    metrics["correct_pixels"] += batch_metrics["correct_pixels"]
    metrics["total_grids"] += batch_metrics["total_grids"]
    metrics["correct_grids"] += batch_metrics["correct_grids"]
    metrics["total_background_pixels"] += batch_metrics["total_background_pixels"]
    metrics["correct_background_pixels"] += batch_metrics["correct_background_pixels"]
    metrics["total_foreground_pixels"] += batch_metrics["total_foreground_pixels"]
    metrics["correct_foreground_pixels"] += batch_metrics["correct_foreground_pixels"]

    metrics["accuracy"] = (metrics["accuracy"] * metrics["num_batches"] + batch_metrics["accuracy"]) / \
                          (metrics["num_batches"] + 1)
    metrics["pixel_accuracy"] = 100. * metrics["correct_pixels"] / metrics["total_pixels"]
    metrics["grid_accuracy"] = 100. * metrics["correct_grids"] / metrics["total_grids"]
    metrics["background_accuracy"] = 100. * metrics["correct_background_pixels"] / \
        metrics["total_background_pixels"] if metrics["total_background_pixels"] > 0 else 0
    metrics["foreground_accuracy"] = 100. * metrics["correct_foreground_pixels"] / \
        metrics["total_foreground_pixels"] if metrics["total_foreground_pixels"] > 0 else 0
    metrics["num_batches"] += 1

    return metrics


def log_metrics_to_wandb(metrics: dict, step: int, type: str = "train"):
    """Logs the metrics to wandb"""
    filtered_metrics = [key for key in metrics.keys() if ("accuracy" in key) or ("loss" in key)]

    if step:
        wandb.log({
            f"{type}_{metric}": metrics[metric] for metric in filtered_metrics
        }, step=step)
    else:
        wandb.log({
            f"{type}_{metric}": metrics[metric] for metric in filtered_metrics
        })
