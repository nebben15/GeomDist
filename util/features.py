import pickle
import json
import numpy as np
import trimesh
from trimesh.triangles import points_to_barycentric
from scipy.spatial import cKDTree

def create_FAUST_index_feautures(txt_path):
    with open(txt_path, 'w') as f:
        f.write("C 6890\n")
        for i in range(6890):
            f.write(f"{i}\n")
    print(f"Successfully wrote FAUST vertex indices to {txt_path}")

def process_vertex_features_from_json_categorical(json_file, txt_path=None, pckl_path=None):
    """
    Converts vertex-feature assignments from a JSON file into categorical encoding.

    Args:
        json_file (str): Path to the JSON file where each feature maps to a list of vertex indices.
        txt_path (str, optional): Path to save the feature array as a .txt file.
        pckl_path (str, optional): Path to save the mapping as a .pckl file.

    Returns:
        np.ndarray: Processed feature array with categorical encoding (integers).
        dict: Mapping of encoding to the original feature.
    """
    # Load the JSON file
    with open(json_file, 'r') as f:
        feature_to_vertices = json.load(f)

    # Invert the mapping: vertex -> feature
    vertex_to_feature = {}
    for feature, vertices in feature_to_vertices.items():
        for vertex in vertices:
            vertex_to_feature[vertex] = feature

    # Extract unique features and create mappings
    unique_features = sorted(set(vertex_to_feature.values()))
    feature_to_id = {feature: idx for idx, feature in enumerate(unique_features)}
    id_to_feature = {idx: feature for feature, idx in feature_to_id.items()}

    # Convert vertex features to categorical encoding
    num_vertices = max(vertex_to_feature.keys()) + 1  # Ensure all vertices are covered
    feature_array = np.zeros((num_vertices, 1), dtype=np.int32)

    for vertex, feature in vertex_to_feature.items():
        feature_id = feature_to_id[feature]
        feature_array[vertex, 0] = feature_id

    # Save the feature array to a .txt file with metadata if txt_path is provided
    if txt_path is not None:
        with open(txt_path, 'w') as f:
            # Write metadata as the first line
            f.write(f"C {len(unique_features) - 1}\n")  # Metadata: Categorical, Max feature value
            # Write the feature array
            np.savetxt(f, feature_array, fmt='%d')
        print(f"Feature array with metadata saved to {txt_path}")

    # Save the mapping to a .pckl file if pckl_path is provided
    if pckl_path is not None:
        with open(pckl_path, 'wb') as f:
            pickle.dump(id_to_feature, f)
        print(f"Mapping saved to {pckl_path}")

    return feature_array, id_to_feature

def process_vertex_features_from_json_one_hot(json_file, txt_path=None, pckl_path=None):
    """
    Converts vertex-feature assignments from a JSON file into one-hot encoding.

    Args:
        json_file (str): Path to the JSON file where each feature maps to a list of vertex indices.
        txt_path (str, optional): Path to save the feature array as a .txt file.
        pckl_path (str, optional): Path to save the mapping as a .pckl file.

    Returns:
        np.ndarray: Processed feature array with one-hot encoding.
        dict: Mapping of encoding to the original feature.
    """
    # Load the JSON file
    with open(json_file, 'r') as f:
        feature_to_vertices = json.load(f)

    # Invert the mapping: vertex -> feature
    vertex_to_feature = {}
    for feature, vertices in feature_to_vertices.items():
        for vertex in vertices:
            vertex_to_feature[vertex] = feature

    # Extract unique features and create mappings
    unique_features = sorted(set(vertex_to_feature.values()))
    feature_to_id = {feature: idx for idx, feature in enumerate(unique_features)}
    id_to_feature = {idx: feature for feature, idx in feature_to_id.items()}

    # Convert vertex features to one-hot encoding
    num_vertices = max(vertex_to_feature.keys()) + 1  # Ensure all vertices are covered
    num_features = len(unique_features)
    feature_array = np.zeros((num_vertices, num_features), dtype=np.int32)

    for vertex, feature in vertex_to_feature.items():
        feature_id = feature_to_id[feature]
        feature_array[vertex, feature_id] = 1

    # Save the feature array to a .txt file with metadata if txt_path is provided
    if txt_path is not None:
        with open(txt_path, 'w') as f:
            # Write metadata as the first line
            f.write(f"O {feature_array.shape[1]}\n")  # Metadata: One-hot, Number of features
            # Write the feature array
            np.savetxt(f, feature_array, fmt='%d')
        print(f"One-hot feature array with metadata saved to {txt_path}")

    # Save the mapping to a .pckl file if pckl_path is provided
    if pckl_path is not None:
        with open(pckl_path, 'wb') as f:
            pickle.dump(id_to_feature, f)
        print(f"Mapping saved to {pckl_path}")

    return feature_array, id_to_feature

def load_features_with_metadata(file_path, expand_dims=True):
    with open(file_path, 'r') as f:
        # Read the first line to determine the format
        first_line = f.readline().strip()
        
        if first_line.startswith('C'):  # Categorical format
            max_feature = int(first_line.split()[1])  # Extract max feature value
            print(f"Categorical format detected. Max feature: {max_feature}")
            
            # Load the rest of the file as numerical features
            features = np.loadtxt(f, dtype=np.int32)
            
            # Validate that all values are within the range [0, max_feature]
            if not np.all((features >= 0) & (features <= max_feature)):
                raise ValueError("Categorical features contain values outside the range [0, max_feature].")
            
            if expand_dims:
                features = np.expand_dims(features, axis=1)

            return features, 'categorical', max_feature
        
        elif first_line.startswith('O'):  # One-hot format
            print("One-hot format detected.")
            
            # Load the rest of the file as one-hot encoded features
            features = np.loadtxt(f, dtype=str)
            features = np.array([list(map(int, line.strip())) for line in features])
            
            # Validate that each row is a valid one-hot vector
            if not np.all((features.sum(axis=1) == 1) & np.all((features == 0) | (features == 1), axis=1)):
                raise ValueError("One-hot features contain invalid rows (not valid one-hot vectors).")
            
            return features, 'one-hot', None
        
        else:
            raise ValueError("Unknown format in the first line of the file.")
        
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