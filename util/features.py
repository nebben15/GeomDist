import pickle
import json
import numpy as np
import trimesh
from trimesh.triangles import points_to_barycentric
from scipy.spatial import cKDTree

def load_features_with_metadata(file_path, expand_dims=True):
    """
    Load features using the new unified header format only.

    Header lines supported:
        - FEATURES_COUNT <int>
        - FEATURES_DIM <int>
        - FEATURES_MAX_ABS <float ... float>

    Returns:
        features: np.ndarray of shape (rows, dim)
        max_meta: np.ndarray of shape (dim,), per-dimension max magnitude (from header if present, else derived)
        dim_out: int, number of feature dimensions (columns)
    """
    with open(file_path, 'r') as f:
        # Read header lines
        first = f.readline().strip()
        header = {"FEATURES_COUNT": None, "FEATURES_DIM": None, "FEATURES_MAX_ABS": None}

        def _try_parse_header(line: str):
            if not line:
                return False
            if line.startswith('FEATURES_COUNT'):
                parts = line.split()
                header["FEATURES_COUNT"] = int(parts[1])
                return True
            if line.startswith('FEATURES_DIM'):
                parts = line.split()
                header["FEATURES_DIM"] = int(parts[1])
                return True
            if line.startswith('FEATURES_MAX_ABS'):
                parts = line.split()
                # parts[0] is label; rest are floats
                header["FEATURES_MAX_ABS"] = np.array([float(p) for p in parts[1:]], dtype=float)
                return True
            return False

        if not _try_parse_header(first):
            raise ValueError("Missing unified header. Expected FEATURES_COUNT/DIM/MAX_ABS lines.")

        # Read additional header lines if present (up to 2 more expected)
        for _ in range(2):
            pos = f.tell()
            ln = f.readline()
            if not ln:
                break
            if not _try_parse_header(ln.strip()):
                # Not a header line; rewind to start of this data line
                f.seek(pos)
                break

        # Load remaining lines as numeric features
        features = np.loadtxt(f, dtype=float)
        if features.ndim == 0:
            features = features.reshape(-1, 1)
        elif features.ndim == 1:
            # If header provides DIM, reshape accordingly; else assume single column
            dim = header["FEATURES_DIM"]
            if dim is not None and features.size % dim == 0:
                features = features.reshape(-1, dim)
            else:
                features = features.reshape(-1, 1)

        # Validate vs header
        if header["FEATURES_COUNT"] is not None and features.shape[0] != header["FEATURES_COUNT"]:
            raise ValueError(f"Feature row count mismatch: header {header['FEATURES_COUNT']} vs data {features.shape[0]}")
        if header["FEATURES_DIM"] is not None and features.shape[1] != header["FEATURES_DIM"]:
            raise ValueError(f"Feature dimension mismatch: header {header['FEATURES_DIM']} vs data {features.shape[1]}")

        max_meta = header["FEATURES_MAX_ABS"]
        if max_meta is None:
            # Derive maxima if not provided
            max_meta = np.max(np.abs(features), axis=0) if features.size else np.zeros((features.shape[1],), dtype=float)

        # Optionally expand dims for 1D inputs (rare with explicit DIM)
        if expand_dims and features.ndim == 1:
            features = np.expand_dims(features, axis=1)

    dim_out = features.shape[1]
    return features, max_meta, dim_out
        
def normalize_features(values: np.ndarray,
                       max_abs_vals: np.ndarray,
                       signed_mask: np.ndarray | None = None,
                       eps: float = 1e-8) -> np.ndarray:
    """Normalize arbitrary feature columns to zero-mean, unit-variance assuming
    a uniform distribution, mirroring the texture normalization used elsewhere.

    For each feature dimension j with maximum magnitude M_j:
      - Unsigned dims assumed in [0, M_j]:    (x / M_j - 0.5) / sqrt(1/12)
      - Signed dims assumed in [-M_j, +M_j]:  (x / (2 M_j)) / sqrt(1/12)

    Parameters
    ----------
    values : np.ndarray, shape (N, D) or (D,)
        Raw feature values to normalize.
    max_abs_vals : np.ndarray, shape (D,)
        Per-dimension max absolute values (M_j). Typically from FEATURES_MAX_ABS.
    signed_mask : np.ndarray | None, shape (D,), optional
        Boolean mask where True marks signed dimensions. If None, it will be
        inferred from `values` by checking if the minimum along each dimension
        is < 0. Note: inferring from a subset may miss negatives; prefer passing
        the mask computed from the full dataset when available.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Normalized features with the same shape as `values`.
    """
    vals = np.asarray(values, dtype=np.float32)
    M = np.asarray(max_abs_vals, dtype=np.float32)
    if vals.ndim == 1:
        vals = vals[None, :]
    if M.ndim != 1 or M.shape[0] != vals.shape[1]:
        raise ValueError(f"max_abs_vals must have shape (D,), got {M.shape} for values shape {vals.shape}")

    denom = np.maximum(M, eps)
    if signed_mask is None:
        # Infer per-dim signedness from the available values (best-effort)
        signed_mask = (np.min(vals, axis=0) < 0)
    else:
        signed_mask = np.asarray(signed_mask, dtype=bool)
        if signed_mask.shape != (vals.shape[1],):
            raise ValueError(f"signed_mask must have shape (D,), got {signed_mask.shape}")

    out = np.empty_like(vals, dtype=np.float32)
    if np.any(signed_mask):
        sm = signed_mask
        out[:, sm] = (vals[:, sm] / (2.0 * denom[sm])) / np.sqrt(1 / 12)
    if np.any(~signed_mask):
        um = ~signed_mask
        out[:, um] = (vals[:, um] / denom[um] - 0.5) / np.sqrt(1 / 12)

    return out if values.ndim > 1 else out[0]

def renormalize_features(norm_values: np.ndarray,
                         max_abs_vals: np.ndarray,
                         signed_mask: np.ndarray | None = None) -> np.ndarray:
    """Inverse of normalize_features.

    Converts normalized features back to their original scale using the same
    signed/unsigned conventions:
      - Unsigned: x = (n * sqrt(1/12) + 0.5) * M
      - Signed:   x = (n * sqrt(1/12)) * (2 M)

    Parameters
    ----------
    norm_values : np.ndarray, shape (N, D) or (D,)
        Normalized features to invert.
    max_abs_vals : np.ndarray, shape (D,)
        Per-dimension max absolute values (M_j).
    signed_mask : np.ndarray | None, shape (D,), optional
        Boolean mask for signed dimensions. If None, all dims are treated as
        unsigned (consistent with prior inference behavior). Pass the exact
        mask used during normalization for perfect symmetry if dims can be
        signed.

    Returns
    -------
    np.ndarray
        De-normalized features, same shape as `norm_values`.
    """
    nvals = np.asarray(norm_values, dtype=np.float32)
    M = np.asarray(max_abs_vals, dtype=np.float32)
    if nvals.ndim == 1:
        nvals = nvals[None, :]
    if M.ndim != 1 or M.shape[0] != nvals.shape[1]:
        raise ValueError(f"max_abs_vals must have shape (D,), got {M.shape} for values shape {nvals.shape}")

    if signed_mask is None:
        signed_mask = np.zeros((nvals.shape[1],), dtype=bool)
    else:
        signed_mask = np.asarray(signed_mask, dtype=bool)
        if signed_mask.shape != (nvals.shape[1],):
            raise ValueError(f"signed_mask must have shape (D,), got {signed_mask.shape}")

    out = np.empty_like(nvals, dtype=np.float32)
    if np.any(signed_mask):
        sm = signed_mask
        out[:, sm] = (nvals[:, sm] * np.sqrt(1 / 12)) * (2.0 * M[sm])
    if np.any(~signed_mask):
        um = ~signed_mask
        out[:, um] = (nvals[:, um] * np.sqrt(1 / 12) + 0.5) * M[um]

    return out if norm_values.ndim > 1 else out[0]
        
def interpolate_features_on_samples(mesh, features, samples, face_indices, interpolation_type="nearest-neighbor"):
    """Interpolate per-vertex features onto sampled points on a mesh surface.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input mesh containing vertices and faces.
    features : np.ndarray, shape (V, d)
        Array of per-vertex features, where V is the number of vertices and d is the feature dimension.
    samples : np.ndarray, shape (N, 3)
        Array of N sampled 3D points on the mesh surface.
    face_indices : np.ndarray, shape (N,)
        Array of face indices indicating which face each sample point lies on.
    interpolation_type : str, optional
        The interpolation method to use. Options are:
            - "nearest-neighbor": Interpolate using barycentric coordinates within the face (default).
            - "barycentric": Assign features from the nearest mesh vertex.

    Returns
    -------
    np.ndarray, shape (N, d)
        Interpolated features for each sampled point.

    Raises
    ------
    ValueError
        If `interpolation_type` is not one of the supported options.

    Notes
    -----
    - For "barycentric", features are interpolated using barycentric coordinates of the sample within its face.
    - For "nearest-neighbor", features are assigned from the nearest mesh vertex to each sample point.
    """

    if interpolation_type == 'nearest-neighbor':
        # Build a KDTree for the mesh vertices
        vertex_tree = cKDTree(mesh.vertices)
        # Query the nearest vertex for each sampled point
        _, nearest_vertex_indices = vertex_tree.query(samples[:, :3])
        # Assign the features of the nearest vertices to the sampled points
        interpolated = features[nearest_vertex_indices]
    elif interpolation_type == 'barycentric':
        # Get triangle vertices for each sample
        triangles = mesh.vertices[mesh.faces[face_indices]]  # shape (N, 3, 3)

        # Compute barycentric coordinates of each sample in its triangle
        bary = points_to_barycentric(triangles, samples)  # shape (N, 3)

        # Gather the features of the triangle vertices
        # shape (N, 3, d)
        tri_vert_feats = features[mesh.faces[face_indices]]

        # Weighted sum along vertex dimension
        # expand bary to (N, 3, 1) so it broadcasts with tri_vert_feats
        interpolated = np.sum(bary[..., np.newaxis] * tri_vert_feats, axis=1)
    else:
        types = ['nearest-neighbor', 'barycentric']
        raise ValueError(f"interpolation_type must be in {types}")

    return interpolated