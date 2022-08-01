import numpy as np, trimesh, sklearn
from sapien.core import Pose
from .o3d_utils import is_o3d, to_o3d, np2mesh
from .trimesh_utils import is_trimesh, to_trimesh
from maniskill2_learn.utils.data import normalize, to_gc, to_nc


def convex_hull(x, o3d=True):
    x = to_trimesh(x)
    x = trimesh.convex.convex_hull(x)
    if o3d:
        x = to_o3d(x)
    return x


def angle(x1, x2):
    if isinstance(x1, np.ndarray):
        x1 = normalize(x1)
        x2 = normalize(x2)
        return np.arccos(np.dot(x1, x2))


def mesh_to_pcd(x, num_points, o3d=True):
    x = to_trimesh(x)
    x = x.sample(num_points)
    if o3d:
        x = to_o3d(x)
    return x


def apply_pose(pose, x):
    if x is None:
        return x
    if isinstance(pose, np.ndarray):
        pose = Pose.from_transformation_matrix(pose)
    assert isinstance(pose, Pose)
    if isinstance(x, Pose):
        return pose * x
    elif isinstance(x, np.ndarray):
        return to_nc(to_gc(x, dim=3) @ pose.to_transformation_matrix().T, dim=3)
    elif is_trimesh(x) or is_o3d(x):
        sign = is_o3d(x)
        x = to_trimesh(x)
        if isinstance(x, trimesh.Trimesh):
            vertices = x.vertices
            faces = x.faces
            vertices = apply_pose(pose, vertices)
            x = trimesh.Trimesh(vertices=vertices, faces=faces)
        elif isinstance(x, trimesh.points.PointCloud):
            vertices = x.vertices
            vertices = apply_pose(pose, vertices)
            x = trimesh.points.PointCloud(vertices=vertices)
        if sign:
            x = to_o3d(x)
        return x
    else:
        print(x, type(x))
        raise NotImplementedError("")


def check_coplanar(vertices):
    pca = sklearn.decomposition.PCA(n_components=3)
    pca.fit(vertices)
    return pca.singular_values_[-1] < 1e-3