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
    pcd_tensor_api= o3d.t.io.read_point_cloud(pcd_path)

    # Optionally move the point cloud to the side
    if side_by_side:
        bbox = mesh.get_axis_aligned_bounding_box()
        offset = bbox.get_extent()[0] * 1.2
        pcd.translate([offset, 0, 0])
    
    # pcd features
    pcd_features = None
    if 'features' in pcd_tensor_api.point:
        pcd_features = pcd_tensor_api.point['features'].numpy()

    # mesh features
    mesh_features = None
    if feature_path is not None:
        mesh_features = np.loadtxt(feature_path, dtype=np.int32)

    # mapping
    mapping = None
    if mapping_path is not None:
        with open(mapping_path, "rb") as f:
            mapping = pickle.load(f)

    # Sanity Checks
    # if features provided -> mapping provided
    assert pcd_features is None or mapping is not None, "pcd-features provided but no mapping provided"
    assert mesh_features is None or mapping is not None, "mesh-features provided but no mapping provided"
    # num feature dims 
    feat_dim = 0
    if mapping is not None:
        unique_features = mapping.keys()
    if pcd_features is not None and mesh_features is not None:
        # feat_dim = pcd_features.shape[1]
        # assert feat_dim == mesh_features.shape[1], "#vertex-featurs != #pcd-features"
        # TODO check mapping features and unique features
        pass
    elif pcd_features is not None:
        # feat_dim = pcd_features.shape[1]
        # TODO check mapping features and unique features\
        pass
    elif mesh_features is not None:
        # feat_dim = mesh_features.shape[1]
        # TODO check mapping features and unique features
        pass


    # Colors
    if (mesh_features is not None or pcd_features is not None):
            colormap = plt.get_cmap("Dark2")
            colors = colormap(np.linspace(0, 1, len(unique_features)))[:, :3]
            feature_to_color = {f: colors[idx] for idx, f in enumerate(unique_features)}
    
    # Mesh coloring
    if mesh_features is not None:
        vertex_colors = np.array([feature_to_color[f] for f in mesh_features])
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    # PCD coloring
    if pcd_features is not None:
        pcd_colors = np.array([
            feature_to_color[round(f)] if round(f) in feature_to_color else [0, 0, 0]
            for f in pcd_features[:, 0]
        ])
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

    # Legend
    print(mapping)

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
    mapping_path = None
    feature_path = None
    ### loong
    # mesh_path = "../shapes/datasets--Zbalpha--shapes/snapshots/56ed38231943963314292f76e9d5bc40ee475f52/loong.obj"
    # pcd_path = "../samples/shapes/loong/self_trained_5_epochs.ply"
    #pcd_path = "../samples/shapes/loong/sample.ply"
    ### spot
    # mesh_path = "../shapes/datasets--Zbalpha--shapes/snapshots/56ed38231943963314292f76e9d5bc40ee475f52/spot/spot_uv_normalized.obj"
    # pcd_path = "../samples/shapes/spot_color/sample.ply"
    ### FAUST
    # mesh_path = "../MPI-FAUST/training/registrations/tr_reg_000.ply"
    # pcd_path = "../samples/FAUST/sample.ply"
    ### FAUST with semantic features
    mesh_path = "../MPI-FAUST/training/registrations/tr_reg_000.ply"
    feature_path = "../SMPL_python_v.1.1.0/smpl_vert_segmentation.txt"
    mapping_path = "../SMPL_python_v.1.1.0/smpl_vert_segmentation_mapping.pkl"
    pcd_path = "../samples/FAUST_features/sample-5.ply"
    visualize_mesh_cloud(mesh_path=mesh_path, pcd_path=pcd_path, feature_path=feature_path, 
                         mapping_path=mapping_path, side_by_side=True)
    