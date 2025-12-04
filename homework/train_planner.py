"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from homework.models import MLPPlanner, TransformerPlanner, save_model
from homework.metrics import PlannerMetric
from homework.datasets.road_dataset import load_data


def train():
    device = torch.device("cpu")  # Force CPU to avoid MPS crash on Mac

    # Load training and validation data (structured inputs only)
    train_loader = load_data("drive_data/train", transform_pipeline="state_only", shuffle=True, batch_size=32)
    val_loader = load_data("drive_data/val", transform_pipeline="state_only", shuffle=False, batch_size=32)

    # Initialize model, optimizer, and masked L1 loss
    model = TransformerPlanner(n_track=10, n_waypoints=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # smaller LR for stability
    loss_fn = nn.L1Loss(reduction="none")  # we'll apply mask manually

    best_val_error = float("inf")

    for epoch in range(45):  # Train longer for better convergence
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            track_left = batch["track_left"].to(device)         # (B, 10, 2)
            track_right = batch["track_right"].to(device)       # (B, 10, 2)
            waypoints = batch["waypoints"].to(device)           # (B, 3, 2)
            mask = batch["waypoints_mask"].to(device)           # (B, 3)

            optimizer.zero_grad()
            preds = model(track_left=track_left, track_right=track_right)

            loss = loss_fn(preds, waypoints)                    # (B, 3, 2)
            loss = loss * mask.unsqueeze(-1)                   # apply mask: (B, 3, 1)
            loss = loss.sum() / mask.sum()                     # average over valid values

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Avg Train Loss = {avg_train_loss:.4f}")

        # ------------------ Validation ------------------
        model.eval()
        metric = PlannerMetric()
        with torch.no_grad():
            for batch in val_loader:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                mask = batch["waypoints_mask"].to(device)

                preds = model(track_left=track_left, track_right=track_right)
                metric.add(preds, waypoints, mask)

        val_scores = metric.compute()
        print(
            f"Validation | L1: {val_scores['l1_error']:.4f}, "
            f"Longitudinal: {val_scores['longitudinal_error']:.4f}, "
            f"Lateral: {val_scores['lateral_error']:.4f}"
        )

        # Save best model
        if val_scores["l1_error"] < best_val_error:
            best_val_error = val_scores["l1_error"]
            save_path = save_model(model)
            print(f"Saved better model to: {save_path}")


if __name__ == "__main__":
    train()
