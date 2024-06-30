"""
Contains training script for PyTorch image classification model.
"""

import typer
import torch
from torchvision.transforms import v2
import data_setup, engine, model_builder, utils
import os

app = typer.Typer()


@app.command()
def train(
    num_epochs: int = typer.Option(50, help="Number of training epochs"),
    batch_size: int = typer.Option(64, help="Batch size for training"),
    learning_rate: float = typer.Option(0.001, help="Learning rate for optimizer"),
    momentum: float = typer.Option(0, help="momentum for sdg optimizer"),
    weight_decay: float = typer.Option(0, help="weight decay for sdg optimizer"),
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
    train_transform = v2.Compose(
        [
            v2.Resize((227, 227)),
            # v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = v2.Compose(
        [
            v2.Resize((227, 227)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create DataLoaders
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=batch_size,
    )

    # Create model
    model = model_builder.AlexNet(num_classes=len(class_names)).to(device)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(
    #     momentum=momentum,
    #     weight_decay=weight_decay,
    #     params=model.parameters(),
    #     lr=learning_rate,
    # )
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=learning_rate,
    )

    # Start training
    engine.one_batch_fit(
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
