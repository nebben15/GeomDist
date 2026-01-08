import argparse
import json
import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree as KDTree


def _parse_scale(scale_arg: str | None) -> float:
    if scale_arg is None:
        return 1.0
    # Accept either a float value or a path to a .npy containing a scalar
    if os.path.isfile(scale_arg):
        val = np.asarray(np.load(scale_arg)).squeeze()
        if val.shape == ():
            return float(val)
        raise ValueError(f"Scale file '{scale_arg}' must contain a single scalar, got shape {val.shape}.")
    try:
        return float(scale_arg)
    except ValueError as e:
        raise ValueError(f"--scale must be a float or a path to a .npy file. Got: {scale_arg}") from e


def _describe_distances(dist: np.ndarray) -> dict:
    # dist is non-negative
    dist = np.asarray(dist)
    return {
        "mean": float(np.mean(dist)),
        "rmse": float(np.sqrt(np.mean(np.square(dist)))),
        "median": float(np.median(dist)),
        "p90": float(np.percentile(dist, 90)),
        "p95": float(np.percentile(dist, 95)),
        "p99": float(np.percentile(dist, 99)),
        "max": float(np.max(dist)),
    }


parser = argparse.ArgumentParser(description="Compute Chamfer distance between two point sets (PLY meshes).")
parser.add_argument('--ply', required=True, type=str, help='Predicted mesh or point cloud (PLY)')
parser.add_argument('--reference', required=True, type=str, help='Reference/ground-truth mesh (PLY)')
parser.add_argument('--scale', type=str, default=None, help='Float scale or path to .npy scalar; defaults to 1.0')
parser.add_argument('--ref-samples', type=int, default=None, help='Number of reference surface samples; default = #pred points')
group = parser.add_mutually_exclusive_group()
group.add_argument('--sample-ref', dest='sample_ref', action='store_true', help='Sample reference mesh surface (default)')
group.add_argument('--no-sample-ref', dest='sample_ref', action='store_false', help='Use reference vertices directly')
parser.set_defaults(sample_ref=True)
parser.add_argument('--seed', type=int, default=None, help='Random seed for sampling')
parser.add_argument('--json', action='store_true', help='Emit results as JSON')
args = parser.parse_args()

scale = _parse_scale(args.scale)

pred_obj = trimesh.load(args.ply)
ref_obj = trimesh.load(args.reference)

if args.seed is not None:
    np.random.seed(args.seed)

# Extract prediction vertices and apply scaling
prediction = pred_obj.vertices * scale

# Determine reference points: sample surface by default
reference_points_info = "vertices"
if args.sample_ref:
    if isinstance(ref_obj, trimesh.Trimesh) and getattr(ref_obj, 'faces', None) is not None and len(ref_obj.faces) > 0:
        n_samples = args.ref_samples if args.ref_samples is not None else int(prediction.shape[0])
        sampled_pts, _ = trimesh.sample.sample_surface(ref_obj, n_samples)
        reference = sampled_pts * scale
        reference_points_info = f"sampled_surface({n_samples})"
    else:
        # Fallback: cannot sample from non-mesh; use vertices
        reference = ref_obj.vertices * scale
        reference_points_info = "vertices (fallback: not a mesh)"
else:
    reference = ref_obj.vertices * scale
    reference_points_info = "vertices (user-disabled sampling)"

# If additional features are present, ignore them for distance computation
if prediction.shape[1] > 3:
    print("Warning: Additional features are present but will not be used in evaluation.")
    prediction = prediction[:, :3]
if reference.shape[1] > 3:
    print("Warning: Additional features are present in the reference but will not be used in evaluation.")
    reference = reference[:, :3]

# KDTree nearest neighbor distances in both directions
tree_pred = KDTree(prediction)
d_ref_to_pred, _ = tree_pred.query(reference)

tree_ref = KDTree(reference)
d_pred_to_ref, _ = tree_ref.query(prediction)

# Directional stats
stats_ref_to_pred = _describe_distances(d_ref_to_pred)
stats_pred_to_ref = _describe_distances(d_pred_to_ref)

# Symmetric Chamfer metrics
cd_l1 = stats_ref_to_pred["mean"] + stats_pred_to_ref["mean"]  # mean L1 in both directions
cd_l2 = float(np.mean(np.square(d_ref_to_pred)) + np.mean(np.square(d_pred_to_ref)))  # mean squared
hausdorff = float(max(np.max(d_ref_to_pred), np.max(d_pred_to_ref)))

result = {
    "points_reference": int(reference.shape[0]),
    "points_prediction": int(prediction.shape[0]),
    "scale": float(scale),
    "reference_points_source": reference_points_info,
    "directional": {
        "reference_to_prediction": stats_ref_to_pred,
        "prediction_to_reference": stats_pred_to_ref,
    },
    "symmetric": {
        "chamfer_L1_mean": cd_l1,
        "chamfer_L2_mean_squared": cd_l2,
        "hausdorff": hausdorff,
    },
}

if args.json:
    print(json.dumps(result, indent=2))
else:
    print("=== Chamfer Distance Report ===")
    print(f"Reference points: {result['points_reference']} | Prediction points: {result['points_prediction']}")
    print(f"Scale applied: {result['scale']}")
    print(f"Reference source: {result['reference_points_source']}")
    print("\nDirectional distances (reference -> prediction):")
    s = result["directional"]["reference_to_prediction"]
    print(f"  mean: {s['mean']:.6f}, rmse: {s['rmse']:.6f}, median: {s['median']:.6f}, p95: {s['p95']:.6f}, max: {s['max']:.6f}")
    print("Directional distances (prediction -> reference):")
    s = result["directional"]["prediction_to_reference"]
    print(f"  mean: {s['mean']:.6f}, rmse: {s['rmse']:.6f}, median: {s['median']:.6f}, p95: {s['p95']:.6f}, max: {s['max']:.6f}")
    print("\nSymmetric metrics:")
    print(f"  Chamfer L1 (mean sum): {result['symmetric']['chamfer_L1_mean']:.6f}")
    print(f"  Chamfer L2 (mean squared sum): {result['symmetric']['chamfer_L2_mean_squared']:.6f}")
    print(f"  Hausdorff (max of directed): {result['symmetric']['hausdorff']:.6f}")