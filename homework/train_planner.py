"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from homework.models import load_model, save_model
from homework.metrics import PlannerMetric
from homework.datasets.road_dataset import load_data


def train(model_name="transformer_planner"):
    print(f"\n==============================")
    print(f"TRAINING MODEL: {model_name}")
    print(f"==============================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader = load_data(
        "drive_data/train",
        transform_pipeline="state_only" if model_name != "cnn_planner" else "image_only",
        shuffle=True,
        batch_size=32,
    )

    val_loader = load_data(
        "drive_data/val",
        transform_pipeline="state_only" if model_name != "cnn_planner" else "image_only",
        shuffle=False,
        batch_size=32,
    )

    # Load correct model
    model = load_model(model_name).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Loss for structured models
    loss_fn = nn.L1Loss(reduction="none")

    best_val_error = float("inf")

    for epoch in range(20):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()

            if model_name == "cnn_planner":
                # CNN uses images instead of lane boundaries
                images = batch["image"].to(device)
                preds = model(images)
                waypoints = batch["waypoints"].to(device)
                mask = batch["waypoints_mask"].to(device)
            else:
                # MLP + Transformer use structured state inputs
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                preds = model(track_left=track_left, track_right=track_right)
                waypoints = batch["waypoints"].to(device)
                mask = batch["waypoints_mask"].to(device)

            loss = loss_fn(preds, waypoints)
            loss = (loss * mask.unsqueeze(-1)).sum() / mask.sum()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Avg Train Loss = {avg_train_loss:.4f}")

        # Validation
        model.eval()
        metric = PlannerMetric()
        with torch.no_grad():
            for batch in val_loader:
                if model_name == "cnn_planner":
                    preds = model(batch["image"].to(device))
                else:
                    preds = model(
                        track_left=batch["track_left"].to(device),
                        track_right=batch["track_right"].to(device),
                    )

                metric.add(preds, batch["waypoints"].to(device), batch["waypoints_mask"].to(device))

        val_scores = metric.compute()
        print(
            f"Validation | L1: {val_scores['l1_error']:.4f}, "
            f"Longitudinal: {val_scores['longitudinal_error']:.4f}, "
            f"Lateral: {val_scores['lateral_error']:.4f}"
        )

        # Save best checkpoint
        if val_scores["l1_error"] < best_val_error:
            best_val_error = val_scores["l1_error"]
            save_path = save_model(model)
            print(f"Saved model to: {save_path}")


if __name__ == "__main__":
    train()
