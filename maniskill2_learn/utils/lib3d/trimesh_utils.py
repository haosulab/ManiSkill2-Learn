import trimesh, numpy as np, open3d as o3d
from maniskill2_learn.utils.data import is_pcd


def to_trimesh(x):
    if is_trimesh(x):
        return x
    elif isinstance(x, np.ndarray):
        assert is_pcd(x)
        return trimesh.points.PointCloud(x)
    elif isinstance(x, o3d.geometry.TriangleMesh):
        vertices = np.asarray(x.vertices)
        faces = np.asarray(x.triangles)
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    elif isinstance(x, o3d.geometry.PointCloud):
        points = np.asarray(x.points)
        return trimesh.points.PointCloud(vertices=points)
    else:
        print(type(x))
        raise NotImplementedError()


def is_trimesh(x):
    return isinstance(x, (trimesh.Trimesh, trimesh.points.PointCloud))
