import torch
from torch.utils.data import DataLoader
from torch.func import functional_call, vmap, grad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
from pathlib import Path
from typing import Callable, Optional, List
import torch.nn.functional as F



class SelfInfluence:
    """
    Computes TracInCP self-influence scores across multiple checkpoints.
    """
    def __init__(self, model, device='cpu', last_layer_only=True):
        self.model = model.to(device)
        self.device = device
        self.last_layer_only = last_layer_only


    def load_checkpoint(self, checkpoint_path):
        """Load model weights from a saved checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location=self.device)['model_state_dict']
        self.model.load_state_dict(state_dict)
        self.model.eval()
        # refresh params from current model weights
        # self.func_model, self.params, self.buffers = make_functional_with_buffers(self.model)

        self.params = dict(self.model.named_parameters())
        self.buffers = dict(self.model.named_buffers())
        if self.last_layer_only:
            self.target_params = {
                k: v for k, v in list(self.params.items())[-2:]
            }
        else:
            self.target_params = self.params


    def _per_sample_grad(self, x, y):
        """Compute flattened gradient vector for a single sample."""
        out = self.model(x.unsqueeze(0))
        loss = F.cross_entropy(out, y.unsqueeze(0), reduction='sum')
        grads = torch.autograd.grad(loss, self.target_params,
                                    retain_graph=False, create_graph=False)
        flat = torch.cat([g.reshape(-1) for g in grads if g is not None])
        return flat

    def compute_batch_influence(self, inputs, labels):
        """
        Compute self-influence for each example in a batch.
        Returns: tensor of shape [B] with self-influence scores.
        """
        self.model.eval()
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        def loss_fn(params, buffers, x, y):
            out = functional_call(self.model, (params, buffers), (x.unsqueeze(0),))
            loss = F.cross_entropy(out, y.unsqueeze(0), reduction='sum')
            return loss

        grad_fn = grad(loss_fn)
        grads = vmap(grad_fn, in_dims=(None, None, 0, 0))(
            self.target_params, self.buffers, inputs, labels
        )

        # flatten gradients and compute squared sum
        if isinstance(grads, dict):
            flat_grads = torch.cat(
                [g.reshape(g.shape[0], -1) for g in grads.values()], dim=1
            )
        else:
            flat_grads = torch.cat([g.reshape(g.shape[0], -1) for g in grads], dim=1)

        influences = (flat_grads ** 2).sum(dim=1)
        return influences.detach().cpu()


    def compute_tracin_self_influence(self, dataloader, checkpoint_paths, eta_list=None):
        """
        Aggregate self-influence across checkpoints (TracInCP).

        Args:
            dataloader: DataLoader over training data.
            checkpoint_paths: list of checkpoint file paths.
            eta_list: optional weighting factors (default = equal).
        Returns:
            tensor [N] of total self-influence scores for training set.
        """
        if eta_list is None:
            eta_list = [1.0 for _ in checkpoint_paths]

        # initialize empty vector for total influence
        num_samples = len(dataloader.dataset)
        total_influence = torch.zeros(num_samples)

        for eta_i, ckpt in zip(eta_list, checkpoint_paths):
            self.load_checkpoint(ckpt)

            offset = 0
            for inputs, labels in tqdm(dataloader, desc=f'Checkpoint {ckpt}'):
                batch_inf = self.compute_batch_influence(inputs, labels)
                total_influence[offset : offset + len(inputs)] += eta_i * batch_inf
                offset += len(inputs)

        return total_influence


class TracInPipeline:
    """
    End-to-end pipeline for computing TracIn-style self-influence.
    
    Args:
        model: trained PyTorch model
        checkpoints: list of checkpoint file paths
        batch_size: evaluation batch size
        preprocessor: optional dataset preprocessor object with .process(dataset)
        collate_fn: optional custom collate function for dataloader
        device: 'cuda' or 'cpu'
    """
    def __init__(
        self,
        model,
        checkpoints: List[str],
        batch_size: int = 128,
        preprocessor: Optional[object] = None,
        collate_fn: Optional[Callable] = None,
        device: str = "cuda"
    ):
        self.model = model
        self.checkpoints = checkpoints
        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.collate_fn = collate_fn
        self.device = device
        self.sic = SelfInfluence(model=self.model, last_layer_only=True, device=device)

    def prepare_dataset(self, dataset):
        """Apply preprocessing if provided."""
        if self.preprocessor is not None:
            dataset = self.preprocessor.process(dataset)
        return dataset

    def make_loader(self, dataset):
        """Create dataloader with custom collate function if given."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def run(self, dataset, plot_results = True, save_path=None):
        """Full pipeline execution: preprocess, load, compute influence."""
        dataset = self.prepare_dataset(dataset)
        dataloader = self.make_loader(dataset)
        print(f"Dataset ready. Computing influence on {len(dataset)} samples...")

        total_influence = self.sic.compute_tracin_self_influence(
            dataloader,
            self.checkpoints
        )
        print("Influence computation complete.")

        # Convert to pandas for analysis
        results = self._assemble_results(dataset, total_influence)

        # Compute metrics and visualize
        metrics = self._compute_metrics(results)
        if plot_results:
            figs = self._generate_plots(results, metrics)
            for name, fig in figs.items():
                fig.show()

        if save_path:
            results.to_csv(f"{save_path}/influence_results.csv", index=False)
            for name, fig in figs.items():
                fig.write_html(f"{save_path}/{name}.html")
            print(f"📁 Results saved to {save_path}")

        return total_influence, results, metrics

    def _assemble_results(self, dataset, influence_scores):
        """Combine influence scores with dataset metadata."""
        results = pd.DataFrame({
            "influence": influence_scores,
        })

        # Add optional metadata
        for col in ["mislabeled", "label", "true_label", "__key__"]:
            if col in dataset.features:
                results[col] = dataset[col]

        results["rank"] = results["influence"].rank(ascending=False)
        results["norm_rank"] = results["rank"] / len(results)
        return results

    def _compute_metrics(self, df):
        """Compute metrics like recall of mislabeled examples."""
        metrics = {}
        if "mislabeled" in df.columns:
            mislabeled = df["mislabeled"].astype(bool)
            total_mislabeled = mislabeled.sum()

            # Recovery (recall) curve: fraction of mislabeled found in top X%
            fractions = np.linspace(0.05, 1.0, 20)
            recovery = []
            precision = []
            for f in fractions:
                k = int(f * len(df))
                top_mislabeled = mislabeled.iloc[df["rank"].nsmallest(k).index].sum()
                recovery.append(top_mislabeled / total_mislabeled)
                precision.append(top_mislabeled / k)

            metrics["fractions"] = fractions
            metrics["recovery"] = recovery
            metrics['precision'] = precision
            metrics["total_mislabeled"] = total_mislabeled
            metrics["auc"] = np.trapz(recovery, fractions)
        return metrics

    # def _generate_plots(self, df, metrics):
    #     figs = {}

    #     # Histogram of self-influence scores
    #     figs["influence_histogram"] = px.histogram(
    #         df, x="influence", nbins=60,
    #         color="mislabeled" if "mislabeled" in df.columns else None,
    #         title="Distribution of Self-Influence Scores"
    #     )

    #     # Recovery curve
    #     if metrics:
    #         figs["recovery_curve"] = go.Figure()
    #         figs["recovery_curve"].add_trace(go.Scatter(
    #             x=metrics["fractions"],
    #             y=metrics["recovery"],
    #             mode="lines+markers",
    #             name="Recovery"
    #         ))
    #         figs["recovery_curve"].update_layout(
    #             title="Mislabeled Recovery Curve",
    #             xaxis_title="Fraction of dataset examined",
    #             yaxis_title="Fraction of mislabeled recovered"
    #         )

    #     # Scatter: influence vs label (optional)
    #     if "label" in df.columns:
    #         figs["influence_by_label"] = px.box(
    #             df, x="label", y="influence",
    #             title="Influence Distribution by Class Label"
    #         )

    #     return figs


    def _generate_plots(self, df, metrics):
        figs = {}

        # --- Histogram of self-influence scores ---
        figs["influence_histogram"] = px.histogram(
            df, x="influence", nbins=60,
            color="mislabeled" if "mislabeled" in df.columns else None,
        )
        figs["influence_histogram"].update_layout(
                title="Distribution of Self-Influence Scores",
                template="plotly_white",
                width=900,
                height=500,
            )

        # --- Recovery curve (recall vs. fraction) ---
        if metrics and "recovery" in metrics:
            figs["recovery_curve"] = go.Figure()
            figs["recovery_curve"].add_trace(go.Scatter(
                x=metrics["fractions"],
                y=metrics["recovery"],
                mode="lines+markers",
                name="Recovery",
                line=dict(color="#1f77b4", width=3)
            ))
            figs["recovery_curve"].update_layout(
                title="Mislabeled Recovery Curve",
                xaxis_title="Fraction of dataset examined",
                yaxis_title="Fraction of mislabeled recovered",
                template="plotly_white",
                width=900,
                height=500,
            )

        # --- Precision vs Recall plot ---
        if metrics and "precision" in metrics and "recovery" in metrics:
            ks = metrics["fractions"]
            precision = metrics["precision"]
            recall = metrics["recovery"]

            figs["precision_vs_recall"] = go.Figure()
            figs["precision_vs_recall"].add_trace(go.Scatter(
                x=recall,
                y=precision,
                mode="lines+markers",
                text=[f"{k*100}%" for k in ks],
                textposition="top center",
                name="Precision vs Recall",
                hovertemplate="Top %{text}<br>Precision=%{y:.3f}<br>Recall=%{x:.3f}<extra></extra>",
                line=dict(width=3, color="#2ca02c"),
                # marker=dict(size=8)
            ))
            figs["precision_vs_recall"].update_layout(
                title="Precision vs Recall Curve (Top-k Ranking)",
                xaxis_title="Recall",
                yaxis_title="Precision",
                template="plotly_white",
                width=900,
                height=500,
            )

        # --- Influence by Label (optional, if available) ---
        if "label" in df.columns:
            figs["influence_by_label"] = px.box(
                df, x="label", y="influence",
            )

            figs["influence_by_label"].update_layout(
                title="Influence Distribution by Class Label",
                xaxis_title="label",
                yaxis_title="influence",
                template="plotly_white",
                width=900,
                height=500,
            )

        return figs
