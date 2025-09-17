import pickle
import json
import numpy as np

def process_vertex_features_from_json(json_file, txt_path=None, pckl_path=None):
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

    # Save the feature array to a .txt file if txt_path is provided
    if txt_path is not None:
        np.savetxt(txt_path, feature_array, fmt='%d')
        print(f"Feature array saved to {txt_path}")

    # Save the mapping to a .pckl file if pckl_path is provided
    if pckl_path is not None:
        with open(pckl_path, 'wb') as f:
            pickle.dump(id_to_feature, f)
        print(f"Mapping saved to {pckl_path}")

    return feature_array, id_to_feature