"""
Contains training script for PyTorch image classification model.
"""

import typer
import torch
from torchvision import transforms
import data_setup, engine, model_builder, utils
import os

app = typer.Typer()


@app.command()
def train(
    num_epochs: int = typer.Option(20, help="Number of training epochs"),
    batch_size: int = typer.Option(64, help="Batch size for training"),
    learning_rate: float = typer.Option(0.01, help="Learning rate for optimizer"),
    momentum: float = typer.Option(0.01, help="momentum for sdg optimizer"),
    weight_decay: float = typer.Option(0.01, help="weight decay for sdg optimizer"),
    train_dir: str = typer.Option(
        "data/pizza_steak_sushi/train", help="Training data directory"
    ),
    test_dir: str = typer.Option(
        "data/pizza_steak_sushi/test", help="Test data directory"
    ),
    model_save_path: str = typer.Option(
        "models/alexnet_model.pth", help="Path to save the trained model"
    ),
):
    """
    Trains a PyTorch image classification model using device-agnostic code.
    """
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create transforms
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    # Create DataLoaders
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=batch_size,
    )

    # Create model
    model = model_builder.AlexNet(num_classes=len(class_names)).to(device)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        momentum=momentum,
        weight_decay=weight_decay,
        params=model.parameters(),
        lr=learning_rate,
    )

    # Start training
    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=num_epochs,
        device=device,
        target_dir=os.path.dirname(model_save_path),
        model_name=os.path.basename(model_save_path),
    )


if __name__ == "__main__":
    app()
