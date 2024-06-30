"""
Contains various utility functions for PyTorch model training and saving.
"""

import torch
from pathlib import Path
import matplotlib.pyplot as plt

plt.style.use("ggplot")


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, target_dir: str, model_name: str, best_valid_loss=float("inf")):
        """
        Save the best model while training.
        Args:
            target_dir (str): directory to save the model.
            model_name (str): model name
            best_valid_loss (_type_, optional): _description_. Defaults to float("inf").
        """
        self.best_valid_loss = best_valid_loss
        # Create target directory
        self.target_dir_path = Path(target_dir)
        self.target_dir_path.mkdir(parents=True, exist_ok=True)
        # Create model save path
        assert model_name.endswith(".pth") or model_name.endswith(
            ".pt"
        ), "model_name should end with '.pt' or '.pth'"
        self.model_name = f"best_{model_name}"
        self.model_save_path = self.target_dir_path / self.model_name

    def __call__(self, current_valid_loss, epoch, model, optimizer, loss_fn):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_fn,
                },
                self.model_save_path,
            )


def save_model(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    target_dir: str,
    model_name: str,
):
    """Saves a PyTorch model to a target directory.

    Args:
    epoch: The end epoch number.
    model: A target PyTorch model to save.
    optimizer: A PyTorch optimizer used to train the model.
    loss_fn: A PyTorch loss function used to train the model.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss_fn,
        },
        model_save_path,
    )


def save_plots(
    train_acc,
    valid_acc,
    train_loss,
    valid_loss,
    target_dir: str = "outputs",
    prefix: str = "",
):
    """
    Function to save the loss and accuracy plots to disk.

    Args:
    train_acc: A list of training accuracies.
    valid_acc: A list of validation accuracies.
    train_loss: A list of training losses.
    valid_loss: A list of validation losses.
    target_dir: A directory for saving the plots to.

    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color="green", linestyle="-", label="train accuracy")
    plt.plot(valid_acc, color="blue", linestyle="-", label="validataion accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    if prefix:
        plt.savefig(f"{target_dir}/{prefix}_accuracy.png")
    else:
        plt.savefig(f"{target_dir}/accuracy.png")

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="orange", linestyle="-", label="train loss")
    plt.plot(valid_loss, color="red", linestyle="-", label="validataion loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    if prefix:
        plt.savefig(f"{target_dir}/{prefix}_loss.png")
    else:
        plt.savefig(f"{target_dir}/loss.png")
