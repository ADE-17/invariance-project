"""
Layer-wise demographic leakage measurement.

Runs linear and MLP probes at each DenseNet block output to trace how
demographic information flows through the network. This reveals whether
DANN/CDANN actually remove demographic information or merely push it
to layers the adversary doesn't reach.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.probes import LinearProbe, MLPProbe, train_probe, evaluate_probe


def extract_layerwise_features(
    backbone: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract features from each DenseNet block for all samples.

    Requires backbone.register_intermediate_hooks() to have been called.

    Args:
        backbone: DenseNet121Backbone with intermediate hooks registered.
        dataloader: DataLoader yielding batch dicts with 'image', 'recorded_sex', 'labels'.
        device: Compute device.

    Returns:
        (layer_features, final_features, demographic_labels, task_labels)
        where layer_features maps layer_name → (N, D_layer) array.
    """
    backbone.eval()
    backbone.to(device)

    layer_features: Dict[str, List[np.ndarray]] = {}
    final_features_list: List[np.ndarray] = []
    demo_labels: List[int] = []
    task_labels_list: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            sexes = batch["recorded_sex"].numpy()

            # Get task labels (use first pathology if labels is a dict)
            if isinstance(batch["labels"], dict):
                first_path = list(batch["labels"].keys())[0]
                task_lab = np.array([float(v) for v in batch["labels"][first_path]])
            else:
                task_lab = batch["labels"].numpy()

            # Forward with intermediate capture
            final, intermediates = backbone.forward_with_intermediates(images)
            final_features_list.append(final.cpu().numpy())

            for layer_name, feats in intermediates.items():
                if layer_name not in layer_features:
                    layer_features[layer_name] = []
                layer_features[layer_name].append(feats.cpu().numpy())

            demo_labels.extend(sexes)
            task_labels_list.extend(task_lab)

    # Concatenate
    result_layers = {
        name: np.concatenate(feat_list, axis=0)
        for name, feat_list in layer_features.items()
    }
    final_arr = np.concatenate(final_features_list, axis=0)
    demo_arr = np.array(demo_labels)
    task_arr = np.array(task_labels_list)

    return result_layers, final_arr, demo_arr, task_arr


def compute_layerwise_leakage(
    backbone: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    probe_type: str = "both",
    mlp_hidden_dims: Tuple[int, int] = (256, 128),
) -> Dict[str, Dict[str, float]]:
    """
    Compute demographic probe AUROC at each network layer.

    This traces how demographic information flows through the encoder
    and whether adversarial training removes it at all layers or only
    at the final layer.

    Args:
        backbone: DenseNet121Backbone (hooks will be registered internally).
        train_loader: Training data loader (for fitting probes).
        test_loader: Test data loader (for evaluating probes).
        device: Compute device.
        probe_type: "linear", "mlp", or "both".
        mlp_hidden_dims: Hidden dimensions for MLP probe.

    Returns:
        Dict mapping layer_name → {linear_auroc, mlp_auroc}.
        Includes "final" for the output layer.
    """
    # Register hooks
    backbone.register_intermediate_hooks()

    try:
        # Extract features from both splits
        train_layers, train_final, train_demo, train_task = extract_layerwise_features(
            backbone, train_loader, device
        )
        test_layers, test_final, test_demo, test_task = extract_layerwise_features(
            backbone, test_loader, device
        )
    finally:
        backbone.remove_hooks()

    results = {}

    # Process each intermediate layer
    all_layers = list(train_layers.keys()) + ["final"]

    for layer_name in all_layers:
        if layer_name == "final":
            X_train, X_test = train_final, test_final
        else:
            X_train = train_layers[layer_name]
            X_test = test_layers[layer_name]

        from src.evaluation.probes import run_all_probes
        probes_res = run_all_probes(
            features_train=X_train, labels_train=train_demo,
            features_test=X_test, labels_test=test_demo,
            task_labels_train=train_task, task_labels_test=test_task,
            feature_dim=X_train.shape[1],
            hidden_dims=mlp_hidden_dims,
            device=str(device)
        )
        
        layer_results = {}
        for probe_name, m in probes_res.items():
            layer_results[f"{probe_name}_auroc"] = m["auroc"]
            layer_results[f"{probe_name}_accuracy"] = m["accuracy"]

        results[layer_name] = layer_results

    return results
