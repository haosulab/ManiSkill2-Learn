import numpy as np, open3d as o3d
from ..lib3d import np2pcd, to_o3d


def visualize_3d(objects, show_frame=True, frame_size=1.0, frame_origin=(0, 0, 0)):
    if not isinstance(objects, (list, tuple)):
        objects = [objects]
    objects = [to_o3d(obj) for obj in objects]
    if show_frame:
        objects.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=frame_origin))
    return o3d.visualization.draw_geometries(objects)


def visualize_pcd(points, colors=None, normals=None, bbox=None, show_frame=False, frame_size=1.0, frame_origin=(0, 0, 0)):
    """Visualize a point cloud."""
    pc = np2pcd(points, colors, normals)
    geometries = [pc]
    if show_frame:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=frame_origin)
        geometries.append(coord_frame)
    if bbox is None:
        bbox = []
    elif not isinstance(bbox, (tuple, list)):
        bbox = [bbox]
    o3d.visualization.draw_geometries(geometries + bbox)
