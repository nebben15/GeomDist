#!/usr/bin/env python3
"""
Export SMPL template vertices (v_template) to a TXT file.

- Input: SMPL model PKL (basicmodel_*_lbs.pkl)
- Output: TXT file with one line per vertex: "x y z" (space-separated, floating point)

Usage:
  python export_smpl_template_vertices.py --input /path/to/basicmodel_neutral_lbs.pkl --output /path/to/smpl_template_vertices_neutral.txt

If --output is omitted, a file next to the input is created with suffix "_vertices.txt".

This script supports neutral/male/female models equally. It reads 'v_template' from the PKL.
"""

import argparse
import os
import pickle
from typing import Any, Dict

try:
    import numpy as np  # optional; script works without NumPy
except Exception:
    np = None


class _Dummy:
    """Fallback type used to satisfy pickle for unknown external classes (e.g., chumpy, scipy)."""
    def __init__(self, *args, **kwargs):
        pass


class _SafeUnpickler(pickle.Unpickler):
    def __init__(self, file_obj):
        # Ensure latin1 decoding for Python2-pickled files
        super().__init__(file_obj, fix_imports=True, encoding="latin1", errors="strict")

    def find_class(self, module, name):
        m = module.lower()
        if m.startswith("chumpy") or m.startswith("scipy"):
            # Return a dummy class to allow unpickling without the actual dependency.
            return _Dummy
        return super().find_class(module, name)


def load_smpl_model(pkl_path: str) -> Dict[str, Any]:
    with open(pkl_path, "rb") as f:
        # Use a safe unpickler to avoid hard deps on chumpy/scipy
        data = _SafeUnpickler(f).load()
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected PKL format: expected dict, got {type(data)}")
    return data


def _shape_2d_len3(x) -> bool:
    """Return True if x looks like shape [N,3] (list/tuple/array)."""
    try:
        return len(x) > 0 and len(x[0]) == 3
    except Exception:
        return False


def export_vertices(v_template, out_path: str) -> None:
    # Validate shape without relying on NumPy
    if not _shape_2d_len3(v_template):
        try:
            shape = (len(v_template), len(v_template[0]) if len(v_template) else 0)
        except Exception:
            shape = "unknown"
        raise ValueError(f"v_template must be of shape [N,3], got {shape}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Write with high precision to preserve exact template values
    with open(out_path, "w", encoding="utf-8") as f:
        for v in v_template:
            f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")


def infer_default_output(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    # Try to infer gender in filename for clarity
    name = os.path.basename(base)
    gender = "neutral"
    for g in ("neutral", "m", "male", "f", "female"):
        if g in name.lower():
            gender = g
            break
    return base + "_vertices.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SMPL v_template to TXT")
    parser.add_argument("--input", required=True, help="Path to SMPL PKL (basicmodel_*_lbs.pkl)")
    parser.add_argument("--output", help="Output TXT path; defaults to <input>_vertices.txt")
    # No external feature file needed; features are the vertices themselves.
    args = parser.parse_args()

    pkl_path = os.path.abspath(args.input)
    out_path = os.path.abspath(args.output) if args.output else infer_default_output(pkl_path)

    model = load_smpl_model(pkl_path)
    # v_template can be under 'v_template' in modern SMPL PKLs
    if "v_template" not in model:
        raise KeyError("SMPL PKL does not contain 'v_template'. Please provide a PKL with template vertices.")
    v_template = model["v_template"]
    # Convert to Python lists if it's a NumPy array
    try:
        if np is not None and isinstance(v_template, np.ndarray):
            v_template = v_template.tolist()
    except Exception:
        pass

    # Compute header information from the vertices themselves (canonical geometric features)
    header_lines = []
    try:
        n_vertices = len(v_template)
        feat_dim = len(v_template[0]) if n_vertices > 0 else 0
        # Per-dimension max magnitude
        max_abs = [0.0] * feat_dim
        for v in v_template:
            for i in range(feat_dim):
                av = abs(float(v[i]))
                if av > max_abs[i]:
                    max_abs[i] = av
        header_lines.append(f"FEATURES_COUNT {n_vertices}")
        header_lines.append(f"FEATURES_DIM {feat_dim}")
        header_lines.append("FEATURES_MAX_ABS " + " ".join(f"{m:.8f}" for m in max_abs))
    except Exception:
        # If anything goes wrong, skip header generation
        header_lines = []

    # Write header (if any) followed by vertices
    if header_lines:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for h in header_lines:
                f.write(h + "\n")
            for v in v_template:
                f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
    else:
        export_vertices(v_template, out_path)
    try:
        n = len(v_template)
    except Exception:
        n = "unknown"
    print(f"Exported {n} vertices to: {out_path}")


if __name__ == "__main__":
    main()
