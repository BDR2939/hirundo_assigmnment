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
from kneed import KneeLocator
from sklearn.mixture import GaussianMixture

colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd']


class Influence:
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
        if isinstance(grads, dict):
            flat = torch.cat([g.reshape(g.shape[0], -1) for g in grads.values()], dim=1)
        else:
            flat = torch.cat([g.reshape(g.shape[0], -1) for g in grads], dim=1)
        return flat.detach().cpu()

    def compute_batch_grads(self, inputs, labels):
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
        return flat_grads.detach().cpu()

        # influences = (flat_grads ** 2).sum(dim=1)
        # return influences.detach().cpu()


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
                batch_grads = self.compute_batch_grads(inputs, labels)
                batch_influences = (batch_grads ** 2).sum(dim=1)
                total_influence[offset : offset + len(inputs)] += eta_i * batch_influences
                offset += len(inputs)

        return total_influence

    # def compute_tracin_influence(
    #     self, train_loader, test_loader, checkpoint_paths, eta_list=None
    #     ):
    #         """
    #         Compute TracIn influence scores between train and test samples.
    #         Self-Influence on the diagonal if train_loader == test_loader (self_influence = np.diag(influence_matrix))

    #         Args:
    #             train_loader: DataLoader over training data.
    #             test_loader: DataLoader over test or validation data.
    #             checkpoint_paths: list of checkpoint paths.
    #             eta_list: list of scalar weights (same length as checkpoints).

    #         Returns:
    #             influence_matrix: np.ndarray of shape [num_train, num_test].
    #         """
    #         if eta_list is None:
    #             eta_list = [1.0 for _ in checkpoint_paths]

    #         num_train = len(train_loader.dataset)
    #         num_test = len(test_loader.dataset)
    #         influence_matrix = np.zeros((num_train, num_test))

    #         # Loop over checkpoints
    #         for eta, ckpt in zip(eta_list, checkpoint_paths):
    #             self.load_checkpoint(ckpt)

    #             # Compute all train gradients
    #             train_grads = []
    #             for x, y in tqdm(train_loader, desc="Train grads"):
    #                 x, y = x.to(self.device), y.to(self.device)
    #                 train_grads.append(self._per_sample_grad(x, y))
    #             train_grads = torch.cat(train_grads, dim=0)

    #             # Compute all test gradients
    #             test_grads = []
    #             for x, y in tqdm(test_loader, desc="Test grads"):
    #                 x, y = x.to(self.device), y.to(self.device)
    #                 test_grads.append(self._per_sample_grad(x, y))
    #             test_grads = torch.cat(test_grads, dim=0)

    #             # Compute dot products
    #             influence = train_grads @ test_grads.T
    #             influence_matrix += eta * influence.numpy()

    #         return influence_matrix

    def compute_tracin_influence(self, train_loader, test_loader, checkpoint_paths, eta_list=None):
        """
            Compute TracIn influence scores between train and test samples.
            Self-Influence on the diagonal if train_loader == test_loader (self_influence = np.diag(influence_matrix))

            Args:
                train_loader: DataLoader over training data.
                test_loader: DataLoader over test or validation data.
                checkpoint_paths: list of checkpoint paths.
                eta_list: list of scalar weights (same length as checkpoints).

            Returns:
                influence_matrix: np.ndarray of shape [num_train, num_test].
            """
        if eta_list is None:
            eta_list = [1.0 for _ in checkpoint_paths]

        num_train = len(train_loader.dataset)
        num_test = len(test_loader.dataset)
        influence_matrix = np.zeros((num_train, num_test))

        for eta, ckpt in zip(eta_list, checkpoint_paths):
            self.load_checkpoint(ckpt)

            # Compute all train gradients batch-wise
            train_grads_list = []
            for inputs, labels in tqdm(train_loader, desc="Train grads"):
                train_grads_list.append(self.compute_batch_grads(inputs, labels))
            train_grads = torch.cat(train_grads_list, dim=0)  # [num_train, D]

            # Compute all test gradients batch-wise
            test_grads_list = []
            for inputs, labels in tqdm(test_loader, desc="Test grads"):
                test_grads_list.append(self.compute_batch_grads(inputs, labels))
            test_grads = torch.cat(test_grads_list, dim=0)  # [num_test, D]

            # Dot product between train and test gradients
            influence_matrix += eta * (train_grads @ test_grads.T).numpy()

        return influence_matrix




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
        mode = "self",
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
        self.mode = mode
        self.engine = Influence(model=self.model, last_layer_only=True, device=device)

        # Choose correct influence computation
        if mode not in ("self", "cross"):
            raise ValueError(f"Unknown mode '{mode}'. Must be 'self' or 'cross'.")

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

    def run(
        self,
        train_dataset,
        test_dataset=None,
        plot_results=True,
        save_path=None
    ):
        """
        Full pipeline execution:
        - Preprocess
        - Compute influence (self or cross)
        - Analyze and optionally visualize
        """
        train_dataset = self.prepare_dataset(train_dataset)
        train_loader = self.make_loader(train_dataset)

        # For self-influence, test_dataset = train_dataset
        if self.mode == "self":
            test_dataset = train_dataset
            print(f"Computing self-influence on {len(train_dataset)} samples")
            total_influence = self.engine.compute_tracin_self_influence(
                dataloader=train_loader,
                checkpoint_paths=self.checkpoints
            )
            print("Self-Influence computation complete.")

            results = self._assemble_results(train_dataset, total_influence)
            metrics = self._compute_metrics(results)

            if plot_results:
                figs = self._generate_plots(results, metrics)
                for fig in figs.values():
                    fig.show()

            if save_path:
                results.to_csv(f"{save_path}/influence_results.csv", index=False)
                for name, fig in figs.items():
                    fig.write_html(f"{save_path}/{name}.html")
                print(f"Results saved to {save_path}")

            return total_influence, results, metrics

        elif self.mode == "cross":
            if test_dataset is None:
                raise ValueError("Test dataset required for cross-sample influence.")
            test_dataset = self.prepare_dataset(test_dataset)
            test_loader = self.make_loader(test_dataset)
            print(f"Computing cross-sample influence: {len(train_dataset)} train → {len(test_dataset)} test")

            influence_matrix = self.engine.compute_tracin_influence(
                train_loader=train_loader,
                test_loader=test_loader,
                checkpoint_paths=self.checkpoints
            )
            print("cross-sample influence computation complete.")

            # You can save and analyze influence matrix directly
            if save_path:
                np.save(f"{save_path}/cross_influence_matrix.npy", influence_matrix)
                print(f"Influence matrix saved to {save_path}/cross_influence_matrix.npy")

            return influence_matrix

    # def run(self, dataset, plot_results = True, save_path=None):
    #     """Full pipeline execution: preprocess, load, compute self-influence."""
    #     dataset = self.prepare_dataset(dataset)
    #     dataloader = self.make_loader(dataset)
    #     print(f"Dataset ready. Computing influence on {len(dataset)} samples...")

    #     total_influence = self.sic.compute_tracin_self_influence(
    #         dataloader,
    #         self.checkpoints
    #     )
    #     print("Influence computation complete.")

    #     # Convert to pandas for analysis
    #     results = self._assemble_results(dataset, total_influence)

    #     # Compute metrics and visualize
    #     metrics = self._compute_metrics(results)
    #     if plot_results:
    #         figs = self._generate_plots(results, metrics)
    #         for name, fig in figs.items():
    #             fig.show()

    #     if save_path:
    #         results.to_csv(f"{save_path}/influence_results.csv", index=False)
    #         for name, fig in figs.items():
    #             fig.write_html(f"{save_path}/{name}.html")
    #         print(f"📁 Results saved to {save_path}")

    #     return total_influence, results, metrics

    def _assemble_results(self, dataset, influence_scores):
        """Combine influence scores with dataset metadata."""
        results = pd.DataFrame({
            "influence": influence_scores,
        })

        # Add optional metadata
        # Check if dataset has HuggingFace features attribute
        if hasattr(dataset, 'features'):
            for col in ["mislabeled", "label", "true_label", "__key__"]:
                if col in dataset.features:
                    results[col] = dataset[col]
        # For PyTorch datasets, try to extract labels if available
        elif hasattr(dataset, '__getitem__') and hasattr(dataset, '__len__'):
            # Try to extract labels from first sample to see if it's a tuple
            try:
                sample = dataset[0]
                if isinstance(sample, tuple) and len(sample) >= 2:
                    # PyTorch dataset format: (data, label, ...)
                    # Extract labels efficiently
                    labels = [dataset[i][1] for i in range(len(dataset))]
                    results["label"] = labels
            except (IndexError, TypeError, AttributeError):
                # If we can't extract labels, just continue without them
                pass

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


    def _generate_plots(self, df, metrics):
        figs = {}

        # --- Histogram of self-influence scores ---
        figs["influence_histogram"] = px.histogram(
            df, x="influence", nbins=60,
            color="mislabeled" if "mislabeled" in df.columns else None,
            color_discrete_sequence=colors[:2] if "mislabeled" in df.columns else [colors[0]],
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
                line=dict(color=colors[0], width=3)
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
                line=dict(width=3, color=colors[3]),
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
            num_labels = df["label"].nunique()
            figs["influence_by_label"] = px.box(
                df, x="label", y="influence", color="label",
                color_discrete_sequence=colors[:num_labels] if num_labels <= len(colors) else colors,
            )

            figs["influence_by_label"].update_layout(
                title="Influence Distribution by Class Label",
                xaxis_title="label",
                yaxis_title="influence",
                template="plotly_white",
                width=900,
                height=500,
                showlegend=False
            )

        return figs
