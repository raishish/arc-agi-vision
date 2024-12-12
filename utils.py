import torch
from torch.utils.data import random_split, Dataset, DataLoader
from torch.nn import functional as F
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from typing import List


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

        if self.transform:
            input_grid = self.transform(input_grid)
            output_grid = self.transform(output_grid)

        return input_grid.unsqueeze(0), output_grid.unsqueeze(0)


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

    if int(val_split) == 0:  # No validation set / Test set
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader, None

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


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    epochs,
    criterion,
    optimizer,
    scheduler=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dataset_variation="ARC-AGI",
    architecture="pixelcnn",
    wandb_project="pixelcnn"
):
    """
    Trains a model on the given dataloaders.

    Args:
        model (torch.nn.Module): model to train
        train_dataloader (torch.utils.data.DataLoader): dataloader for training
        val_dataloader (torch.utils.data.DataLoader): dataloader for validation/testing
        epochs (int): number of epochs to train the model
        criterion (torch.nn.Module): loss function
        optimizer (torch.optim.Optimizer): optimizer
        scheduler (torch.optim.lr_scheduler, optional): learning rate scheduler
        device (torch.device, optional): device to use for training
        dataset_variation (str, optional): variation of the dataset (for logging)
        architecture (str, optional): architecture of the model (for logging)
        wandb_project (str, optional): wandb project name (for wandb support)
    """
    # Initialize wandb
    wandb.init(project=wandb_project, entity="raishish")
    wandb.config.update({
        "architecture": architecture,
        "dataset": dataset_variation,
        "epochs": epochs,
        "batch_size": train_dataloader.batch_size,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

    learning_rate = optimizer.param_groups[0]['lr']  # Save the learning rate
    model.to(device)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train_pixels = 0
        total_train_pixels = 0
        total_train_grids = 0
        correct_train_grids = 0
        total_train_background_pixels = 0
        correct_train_background_pixels = 0
        total_train_foreground_pixels = 0
        correct_train_foreground_pixels = 0

        with tqdm(total=len(train_dataloader), desc=f"Train Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for inputs, targets in train_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.squeeze(1)  # Remove the channel dimension from targets
                # forward pass
                outputs = model(inputs)
                # compute loss
                loss = criterion(outputs, targets.long())  # Convert targets to Long
                train_loss += loss.item()
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calculate accuracy metrics
                metrics = get_accuracy_metrics(outputs, targets)
                total_train_pixels += metrics['total_pixels']
                correct_train_pixels += metrics['correct_pixels']
                total_train_grids += metrics['total_grids']
                correct_train_grids += metrics['correct_grids']
                total_train_background_pixels += metrics['total_background_pixels']
                correct_train_background_pixels += metrics['correct_background_pixels']
                total_train_foreground_pixels += metrics['total_foreground_pixels']
                correct_train_foreground_pixels += metrics['correct_foreground_pixels']

                # Update progress bar
                pbar.set_postfix({
                    'CE Loss': train_loss / (pbar.n + 1),
                    'Pixel accuracy': f"{100. * metrics['pixel_accuracy']:.2f} %",
                    'Grid accuracy': f"{100. * metrics['grid_accuracy']:.2f} %",
                    'Background accuracy': f"{100. * metrics['background_accuracy']:.2f} %",
                    'Foreground accuracy': f"{100. * metrics['foreground_accuracy']:.2f} %"
                })
                pbar.update(1)

        # Per epoch metrics
        avg_train_loss = train_loss / len(train_dataloader)
        train_pixel_accuracy = 100. * correct_train_pixels / total_train_pixels
        train_grid_accuracy = 100. * correct_train_grids / total_train_grids
        train_background_accuracy = 100. * correct_train_background_pixels / total_train_background_pixels
        train_foreground_accuracy = 100. * correct_train_foreground_pixels / total_train_foreground_pixels

        # Validation phase
        model.eval()
        val_loss = 0.0
        total_val_pixels = 0
        correct_val_pixels = 0
        total_val_grids = 0
        correct_val_grids = 0
        total_val_background_pixels = 0
        correct_val_background_pixels = 0
        total_val_foreground_pixels = 0
        correct_val_foreground_pixels = 0

        with torch.no_grad():
            with tqdm(total=len(val_dataloader), desc=f"Validation Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
                for inputs, targets in val_dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    targets = targets.squeeze(1)  # Remove the channel dimension from targets
                    outputs = model(inputs)
                    loss = criterion(outputs, targets.squeeze(1).long())
                    val_loss += loss.item()

                    # Calculate accuracy metrics
                    metrics = get_accuracy_metrics(outputs, targets)
                    total_val_pixels += metrics['total_pixels']
                    correct_val_pixels += metrics['correct_pixels']
                    total_val_grids += metrics['total_grids']
                    correct_val_grids += metrics['correct_grids']
                    total_val_background_pixels += metrics['total_background_pixels']
                    correct_val_background_pixels += metrics['correct_background_pixels']
                    total_val_foreground_pixels += metrics['total_foreground_pixels']
                    correct_val_foreground_pixels += metrics['correct_foreground_pixels']

                    # Update progress bar
                    pbar.set_postfix({
                        'CE Loss': val_loss / (pbar.n + 1),
                        'Pixel Accuracy': f"{100. * metrics['pixel_accuracy']:.2f} %",
                        'Grid accuracy': f"{100. * metrics['grid_accuracy']:.2f} %",
                        'Background accuracy': f"{100. * metrics['background_accuracy']:.2f} %",
                        'Foreground accuracy': f"{100. * metrics['foreground_accuracy']:.2f} %"
                    })
                    pbar.update(1)

        # Per epoch metrics
        avg_val_loss = val_loss / len(val_dataloader)
        val_pixel_accuracy = 100. * correct_val_pixels / total_val_pixels
        val_grid_accuracy = 100. * correct_val_grids / total_val_grids
        val_background_accuracy = 100. * correct_val_background_pixels / total_val_background_pixels
        val_foreground_accuracy = 100. * correct_val_foreground_pixels / total_val_foreground_pixels

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_pixel_accuracy": train_pixel_accuracy,
            "train_grid_accuracy": train_grid_accuracy,
            "train_background_accuracy": train_background_accuracy,
            "train_foreground_accuracy": train_foreground_accuracy,
            "val_loss": avg_val_loss,
            "val_pixel_accuracy": val_pixel_accuracy,
            "val_grid_accuracy": val_grid_accuracy,
            "val_background_accuracy": val_background_accuracy,
            "val_foreground_accuracy": val_foreground_accuracy,
        })

        # Step the scheduler if provided
        if scheduler:
            scheduler.step(avg_val_loss)

    # Save the model
    torch.save(model.state_dict(), f"{architecture}_{epochs}epochs_lr{learning_rate}.pth")
    wandb.save(f"{architecture}_{epochs}epochs_lr{learning_rate}.pth")

    # Finish the wandb run
    wandb.finish()


def plot_batch(models: List[torch.nn.Module], inputs: torch.Tensor, targets: torch.Tensor):
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

    for i, (input_grid, target_grid, predicted_grid) in enumerate(zip(inputs, targets, predicted_grids)):
        axs[i][0].imshow(input_grid.squeeze().numpy())
        axs[i][0].set_title('Input Grid')
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
