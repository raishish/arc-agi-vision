#! /usr/bin/env python3

import os
import argparse
import torch
from torch import nn
from time import time
from datetime import timedelta
from torchmetrics.classification import MulticlassJaccardIndex
from torch.optim import Adam
import pandas as pd

from utils import get_dataloaders, PadTransform, get_class_weights
from segmenter import VisionTransformer, MaskTransformer, Segmenter
from segmenter import _process_one_batch


def train(
    model: torch.nn.Module,
    device: torch.device,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    num_epochs: int,
    optimizer,
    loss_criterion,
    acc_criterion=None,
    patience: int = 10,
    save_dir: str = os.path.abspath("."),
    checkpoint_path: str = None,
    verbose: int = 0
):
    """
    Trains the model

    Args:
        model (torch.nn.module): pytorch model
        device (torch.device): Device for training
        train_dataloader (torch.Dataloader): Train dataloader
        val_dataloader (torch.Dataloader): Validation dataloader
        num_epochs (int): number of epochs to train
        loss_criterion (nn.modules.loss): Loss criterion
        acc_criterion (): Accuracy criterion
        optimizer (torch.optim): Optimizer
        patience (int): Number of epochs to wait for val_acc to improve
                        before breaking training loop
        save_dir (str): location to save model if val_acc improves
        checkpoint_path (str): location to load saved model from, it if exists
        verbose (int): verbosity of logs
    """
    model.to(device)
    best_val_acc = 0
    counter = 0  # Counter for early stopping
    epoch_begin = 0
    num_train_batches = len(train_dataloader)
    num_val_batches = len(val_dataloader)

    if checkpoint_path and os.path.exists(os.path.abspath(checkpoint_path)):
        checkpoint_path = os.path.abspath(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_begin = checkpoint['epoch'] + 1   # epoch to start training with
        best_val_acc = checkpoint['best_val_acc']
        print(f"Model resumed from checkpoint at {checkpoint_path}")

    for epoch in range(epoch_begin, num_epochs):
        t1 = time()
        if verbose > 0:
            print("=" * 25, f"Epoch {epoch + 1}", "=" * 25, "\n")
        model.train()
        total_train_acc = 0
        total_train_loss = 0

        for batch_num, data in enumerate(train_dataloader):
            batch_loss, batch_acc = _process_one_batch(
                data, batch_num, model, device, "train",
                optimizer, loss_criterion, acc_criterion=acc_criterion,
                verbose=verbose
            )

            # Aggregate for epoch level reporting
            total_train_loss += batch_loss
            total_train_acc += batch_acc

        elapsed = str(timedelta(seconds=time() - t1))
        if verbose > 0:
            print(f"\nProcessed training epoch {epoch + 1} in {elapsed}")
            print("-" * 60, "\n")

        avg_train_loss = total_train_loss / num_train_batches
        avg_train_acc = total_train_acc / num_train_batches

        # Validate the model
        t2 = time()
        if verbose > 0:
            print("\n", "-" * 25, "Validation", "-" * 25, "\n")
        model.eval()
        total_val_acc = 0
        total_val_loss = 0
        with torch.no_grad():
            for batch_num, data in enumerate(val_dataloader):
                batch_loss, batch_acc = _process_one_batch(
                    data, batch_num, model, device, "eval",
                    optimizer, loss_criterion, acc_criterion=acc_criterion,
                    verbose=verbose
                )

                total_val_loss += batch_loss
                total_val_acc += batch_acc

        avg_val_loss = total_val_loss / num_val_batches
        avg_val_acc = total_val_acc / num_val_batches

        elapsed = str(timedelta(seconds=time() - t2))
        if verbose > 0:
            print(f"\nProcessed validation epoch {epoch + 1} in {elapsed}\n")

        # Save the best model
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            counter = 0  # Reset the counter when there's an improvement
            best_model_path = os.path.join(
                save_dir, "best_segmenter_model.pth"
            )
            checkpoint = {
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            best_checkpoint_path = os.path.join(
                save_dir, "best_segmenter_checkpoint.pth"
            )
            print(
                f"Found better validation accuracy ({best_val_acc})",
                f"saving model to {best_model_path}",
                f"saving checkpoint to {best_checkpoint_path}"
            )

            torch.save(model.state_dict(), best_model_path)
            torch.save(checkpoint, best_checkpoint_path)
        else:
            counter += 1

        print(
            f"\nEpoch {epoch + 1} -",
            f"Training loss: {avg_train_loss:.4f} -",
            f"Training Accuracy (mIOU): {avg_train_acc * 100:.2f}% -",
            f"Validation loss: {avg_val_loss:.4f} -",
            f"Validation Accuracy (mIOU): {avg_val_acc * 100:.2f}% -",
            f"Best Validation Accuracy (mIOU): {best_val_acc * 100:.2f}%"
        )
        print("=" * 60, "\n")

        # Early stopping condition
        if counter >= patience:
            print(
                f"No improvement in validation accuracy for {patience}",
                "epochs. Early stopping..."
            )
            break


def get_commandline_args():
    """Get commandline arguments

    Returns:
        argparse.Namespace: a dict-type object to access arguments
    """
    def is_valid_path(parser, arg):
        """Checks if the passed argument is a valid file / directory"""
        if not os.path.exists(arg):
            parser.ArgumentTypeError(
                "The passed directory / file %s does not exist!" % arg
            )
        else:
            return os.path.abspath(arg)     # return absolute path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        help="dataset directory, contains (train, val, unlabeled dirs)",
        type=lambda x: is_valid_path(parser, x),
        required=True
    )
    parser.add_argument(
        "--save-dir",
        "-s",
        help="Directory to save models / checkpoints",
        type=lambda x: is_valid_path(parser, x),
        required=False
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        help="path to model checkpoint to resume training from",
        type=lambda x: is_valid_path(parser, x),
        required=False
    )
    parser.add_argument(
        "--batchsize",
        "-b",
        help="Batchsize for training",
        type=int,
        required=False
    )
    parser.add_argument(
        "--epochs",
        "-e",
        help="Number of epochs for training",
        type=int,
        required=False
    )
    parser.add_argument(
        "--output-file",
        "-o",
        help="stdout file",
        required=False
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity (-v, -vv, etc)",
        required=False
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_commandline_args()
    dataset_dir = args.dataset
    save_dir = args.save_dir
    checkpoint_path = args.checkpoint
    verbose = args.verbose

    train_data = pd.read_csv("data/train_small_grids.csv")

    transform = PadTransform((10, 10))
    if args.batchsize:
        batch_size = args.batchsize
    else:
        batch_size = 32

    train_dataloader, val_dataloader = get_dataloaders(
        train_data,
        val_split=0.2,
        batch_size=batch_size,
        transform=PadTransform((10, 10)),
        shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model params
    num_classes = 10
    im_size = (10, 10)
    patch_size = 2
    in_channels = 1

    # Encoder params
    encoder_embed_dim = 48
    encoder_mlp_dim = 192
    encoder_num_heads = 3
    encoder_num_layers = 6
    encoder_dropout = 0.01

    # Decoder params
    decoder_embed_dim = 48
    decoder_mlp_dim = 192
    decoder_num_heads = 3
    decoder_num_layers = 6
    decoder_dropout = 0.01

    # Initialize the model
    encoder = VisionTransformer(
        im_size,
        patch_size,
        encoder_num_layers,
        encoder_embed_dim,
        encoder_mlp_dim,
        encoder_num_heads,
        num_classes,
        dropout=encoder_dropout,
        channels=in_channels
    )
    decoder = MaskTransformer(
        num_classes,
        patch_size,
        encoder_embed_dim,
        decoder_num_layers,
        decoder_num_heads,
        decoder_embed_dim,
        decoder_mlp_dim,
        dropout=decoder_dropout
    )
    model = Segmenter(encoder, decoder, num_classes)
    print("Model instantiated.", flush=True)

    # Training params
    # Pixel-level multi-class classification task
    class_weights = get_class_weights(train_dataloader)
    loss_criterion = nn.CrossEntropyLoss(weight=class_weights)
    jaccard = MulticlassJaccardIndex(num_classes=10).to(device)   # Accuracy

    if args.epochs:
        num_epochs = args.epochs
    else:
        num_epochs = 50

    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=6, threshold=0.001, factor=0.1
    )
    patience = 10  # Number of epochs to wait for improvement

    train(
        model, device,
        train_dataloader, val_dataloader,
        num_epochs,
        optimizer,
        loss_criterion, acc_criterion=jaccard,
        patience=patience,
        save_dir=save_dir, checkpoint_path=checkpoint_path,
        verbose=verbose
    )
