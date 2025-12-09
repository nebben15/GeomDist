import argparse
import trimesh
import numpy as np

def sample_surface_points(mesh_path, num_points, output_path):
    mesh = trimesh.load(mesh_path)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Loaded mesh is not a trimesh.Trimesh object.")

    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    # Create a point cloud
    cloud = trimesh.points.PointCloud(points)
    # Export as PLY
    cloud.export(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample points from mesh surface and save as .ply")
    parser.add_argument("--mesh_path", type=str, help="Path to input mesh file")
    parser.add_argument("--num_points", type=int, help="Number of points to sample")
    parser.add_argument("--output_path", type=str, help="Path to output .ply file")
    args = parser.parse_args()

    sample_surface_points(args.mesh_path, args.num_points, args.output_path)