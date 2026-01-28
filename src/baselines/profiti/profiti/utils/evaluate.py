import logging
from typing import Tuple

import torch
import torch.nn as nn


class Evaluator:
    """Evaluator class for ProFITi model."""

    def __init__(
        self,
        model: nn.Module,
        args,
        device: torch.device,
    ):
        self.model = model
        self.device = device
        self.args = args

    def evaluate(self, data_loader) -> float:
        """Evaluate model on given data loader."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                batch = [tensor.to(self.device) for tensor in batch]
                tx, cx, x, mx, tq, cq, y, mq = batch

                self.model.distribution(tx, cx, x, mx, tq, cq, mq)
                if self.args.marginal_training:
                    mnll = self.model.compute_mnll(y, mq)
                    current_queries = mq.sum()
                    total_loss += mnll.item()  # mnll is already summed for all the variables
                    total_samples += current_queries
                else:
                    njnll = self.model.compute_njnll(y, mq)
                    total_loss += njnll.sum().item()
                    total_samples += tx.shape[0]

        return total_loss / total_samples

    def evaluate_additional_metrics(self, data_loader, nsamples: int = 1000, chunks=10) -> Tuple[float, float]:
        """Evaluate additional metrics (MSE, Robust MSE, CRPS, Energy Score) on given data loader.
        This function also collects all samples, targets, and masks for further analysis.
        """
        self.model.eval()
        total_mse = 0.0
        total_robust_mse = 0.0
        total_crps = 0.0
        total_energy_score = 0.0
        total_samples = 0
        total_queries = 0
        all_samples = []
        all_targets = []
        all_masks = []

        with torch.no_grad():
            for batch in data_loader:
                batch = [tensor.to(self.device) for tensor in batch]
                tx, cx, x, mx, tq, cq, y, mq = batch
                if not mq.any():
                    continue

                self.model.distribution(tx, cx, x, mx, tq, cq, mq)
                samples, _ = self.model.samples(mq, nsamples=nsamples)  # unused log_det
                samples = samples.clamp(-10, 10)
                mse = self.model.mse(y, mq, nsamples=100)
                robust_mse = self.model.robust_mse(y, mq, nsamples=100)
                crps = self.model.crps(y, mq, nsamples=nsamples)
                energy_score = self.model.energy_score(y, mq, nsamples=nsamples)

                current_queries = mq.sum().item()
                current_instances = tx.shape[0]
                total_mse += mse * current_queries
                total_robust_mse += robust_mse * current_queries
                total_crps += crps * current_queries
                total_energy_score += energy_score * current_instances
                total_queries += current_queries
                total_samples += current_instances

                all_samples.append(samples.cpu())
                all_masks.append(mq.cpu())
                all_targets.append(y.cpu())
            return (
                total_mse / total_queries,
                total_robust_mse / total_queries,
                total_crps / total_queries,
                total_energy_score / total_samples,
                all_samples,
                all_targets,
                all_masks,
            )

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        return checkpoint
