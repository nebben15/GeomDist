import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pickle

def visualize_mesh_cloud(mesh_path, pcd_path, feature_path=None,
                         mapping_path=None, side_by_side=True):
    """
    Visualize a mesh and point cloud, with optional feature-based coloring.

    Args:
        mesh_path (str): Path to the mesh file.
        pcd_path (str): Path to the point cloud file.
        feature_path (str, optional): Path to the .txt file containing feature vectors for the mesh.
        mapping_path (str, optional): Path to the .pkl file mapping feature IDs to class names.
        side_by_side (bool): Whether to display the point cloud next to the mesh.
    """
    # Initialize the GUI app
    o3d.visualization.gui.Application.instance.initialize()
    
    # Load mesh and point cloud
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    # Optionally move the point cloud to the side
    if side_by_side:
        bbox = mesh.get_axis_aligned_bounding_box()
        offset = bbox.get_extent()[0] * 1.2
        pcd.translate([offset, 0, 0])
    
    # Legend container
    legend_colors = {}

    # Apply feature-based coloring if feature_path is provided
    if feature_path is not None:
        # Load the feature file
        features = np.loadtxt(feature_path, dtype=np.float32)
        
        # Ensure the number of features matches the number of vertices
        if features.shape[0] != np.asarray(mesh.vertices).shape[0]:
            raise ValueError("Number of features does not match the number of vertices in the mesh.")
        
        # Load mapping if provided
        feature_mapping = None
        if mapping_path is not None:
            with open(mapping_path, "rb") as f:
                feature_mapping = pickle.load(f)  # dict[int -> str]
        
        # Handle categorical encoding
        if features.ndim == 1:
            unique_features = np.unique(features.astype(int))
            colormap = plt.get_cmap("Dark2")
            colors = colormap(np.linspace(0, 1, len(unique_features)))[:, :3]
            feature_to_color = {f: colors[idx] for idx, f in enumerate(unique_features)}
            vertex_colors = np.array([feature_to_color[int(f)] for f in features])
            
            # Build legend using mapping if available
            for f, c in feature_to_color.items():
                label = feature_mapping[f] if feature_mapping and f in feature_mapping else str(f)
                legend_colors[label] = c
        
        elif features.ndim == 2:  # One-hot encoding
            unique_features = np.unique(features, axis=0)
            colormap = plt.get_cmap("Dark2")
            colors = colormap(np.linspace(0, 1, len(unique_features)))[:, :3]
            feature_to_color = {tuple(f): colors[idx] for idx, f in enumerate(unique_features)}
            vertex_colors = np.array([feature_to_color[tuple(f)] for f in features])
            
            # Build legend (not typical for SMPL, but handled anyway)
            for f, c in feature_to_color.items():
                label = str(f)
                legend_colors[label] = c
        else:
            raise ValueError("Invalid feature format. Must be 1D (categorical) or 2D (one-hot).")
        
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    # Process point cloud features
    pcd_features = None
    print(np.asarray(pcd.points).shape)
    if np.asarray(pcd.points).shape[1] > 3:
        pcd_data = np.asarray(pcd.points)
        pcd_points, pcd_features = pcd_data[:, :3], pcd_data[:, 3:]
        pcd.points = o3d.utility.Vector3dVector(pcd_points)

    # Apply feature-based coloring to the point cloud if mapping matches
    if pcd_features is not None and feature_path is not None:
        if pcd_features.shape[1] != features.shape[1]:
            raise ValueError("Feature dimensions of the point cloud do not match the mesh features.")

        if features.ndim == 1:  # Categorical encoding
            pcd_colors = np.array([feature_to_color[int(f)] for f in pcd_features[:, 0]])
        elif features.ndim == 2:  # One-hot encoding
            pcd_colors = np.array([feature_to_color[tuple(f)] for f in pcd_features])
        else:
            raise ValueError("Invalid feature format for point cloud. Must be 1D (categorical) or 2D (one-hot).")

        pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

    # Create the visualizer window
    vis = o3d.visualization.O3DVisualizer("Mesh + PointCloud", 1024, 768)
    vis.add_geometry("mesh", mesh)
    vis.add_geometry("pointcloud", pcd)
    vis.show_settings = True  # makes right-hand menu visible
        
    # Run the app
    app = o3d.visualization.gui.Application.instance
    app.add_window(vis)
    app.run()

if __name__ == "__main__":
    #mesh_path = "../shapes/datasets--Zbalpha--shapes/snapshots/56ed38231943963314292f76e9d5bc40ee475f52/loong.obj"
    #mesh_path = "../MPI-FAUST/training/registrations/tr_reg_000.ply"
    #pcd_path = "../samples/shapes/FAUST/sample.ply"
    ### spot
#     mesh_path = "/home/ben/Thesis/shapes/datasets--Zbalpha--shapes/snapshots/" \
# "56ed38231943963314292f76e9d5bc40ee475f52/spot/spot_uv_normalized.obj"
#     pcd_path = "../samples/shapes/spot_color/sample.ply"
    ### FAUST with semantic features
    mesh_path = "../MPI-FAUST/training/registrations/tr_reg_000.ply"
    feature_path = "../SMPL_python_v.1.1.0/smpl_vert_segmentation.txt"
    mapping_path = "../SMPL_python_v.1.1.0/smpl_vert_segmentation_mapping.pkl"
    pcd_path = "../samples/FAUST_features/sample.ply"
    visualize_mesh_cloud(mesh_path=mesh_path, pcd_path=pcd_path, feature_path=feature_path, 
                         mapping_path=mapping_path, side_by_side=True)
