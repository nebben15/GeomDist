import open3d as o3d

def visualize_mesh_cloud(mesh_path, pcd_path, side_by_side=True):
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
    
    # Create the visualizer window
    vis = o3d.visualization.O3DVisualizer("Mesh + PointCloud", 1024, 768)
    vis.add_geometry("mesh", mesh)
    vis.add_geometry("pointcloud", pcd)
    vis.show_settings = True
    
    # Run the app
    app = o3d.visualization.gui.Application.instance
    app.add_window(vis)
    app.run()(vis)
    app.run()

if __name__ == "__main__":
    #mesh_path = "../shapes/datasets--Zbalpha--shapes/snapshots/56ed38231943963314292f76e9d5bc40ee475f52/loong.obj"
    mesh_path = "../MPI-FAUST/training/registrations/tr_reg_000.ply"
    pcd_path = "../samples/shapes/FAUST/sample.ply"
    visualize_mesh_cloud(mesh_path=mesh_path, pcd_path=pcd_path, side_by_side=False)
