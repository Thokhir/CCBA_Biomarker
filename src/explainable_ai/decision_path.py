"""Decision path / feature-split analysis for the Random Forest ensemble.

The fitted model's 500 trees are unusually shallow (depth 1-2, confirmed by
direct inspection) - a consequence of class_weight="balanced" combined with
bootstrap resampling of a small, imbalanced 44-sample training set (35
tumor / 9 normal): many bootstrap draws contain very few unique minority
samples, so trees reach purity almost immediately. This makes individual
trees fully renderable without truncation, and is itself a notable, real
modeling characteristic worth documenting - the ensemble behaves as bagged
near-stumps, with predictive power coming from averaging many such shallow
learners rather than from any single deep tree.
"""
import numpy as np


def summarize_tree_depths(model) -> dict:
    depths = [est.get_depth() for est in model.estimators_]
    leaves = [est.get_n_leaves() for est in model.estimators_]
    return {
        "n_estimators": len(model.estimators_),
        "depth_min": int(min(depths)),
        "depth_mean": float(np.mean(depths)),
        "depth_max": int(max(depths)),
        "leaves_min": int(min(leaves)),
        "leaves_mean": float(np.mean(leaves)),
        "leaves_max": int(max(leaves)),
    }


def select_representative_tree_indices(model, n: int = 3) -> list:
    """Picks a small, illustrative spread of trees: the deepest, the
    shallowest, and one from the middle of the depth distribution.
    """
    depths = [est.get_depth() for est in model.estimators_]
    order = np.argsort(depths)
    indices = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    return sorted(set(indices))[:n]
