import numpy as np
import open3d as o3d
import trimesh
from maniskill2_learn.utils.data import is_pcd


def is_o3d(x):
    return isinstance(x, (o3d.geometry.TriangleMesh, o3d.geometry.PointCloud, o3d.geometry.OrientedBoundingBox, o3d.geometry.AxisAlignedBoundingBox))


def to_o3d(x):
    """
    Numpy support is for pcd!
    """
    if is_o3d(x):
        return x
    elif isinstance(x, np.ndarray):
        assert is_pcd(x)
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(x))
    elif isinstance(x, trimesh.Trimesh):
        return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(x.vertices), o3d.utility.Vector3iVector(x.faces))
    elif isinstance(x, trimesh.points.PointCloud):
        return o3d.geometry.PointCloud(x.vertices)
    else:
        print(type(x))
        raise NotImplementedError()


def one_point_vis(x, num=100, noise=0.02, colors=[1, 0, 0]):
    noise_vector = np.random.randn(num, 3) * noise
    colors = np.repeat(np.array(colors)[None, :], len(x), axis=0)
    x = x + noise_vector
    return np2pcd(x)


# ---------------------------------------------------------------------------- #
# Convert in opne3d
# ---------------------------------------------------------------------------- #


def merge_mesh(meshes):
    if not isinstance(meshes, (list, tuple)):
        return meshes
    if len(meshes) == 0:
        return None
    # Merge without color and normal
    vertices = np.zeros((0, 3))
    triangles = np.zeros((0, 3))

    for mesh in meshes:
        if mesh is None:
            continue
        vertices_i = np.asarray(mesh.vertices).copy()
        triangles_i = np.asarray(mesh.triangles).copy()
        triangles_i += vertices.shape[0]
        vertices = np.append(vertices, vertices_i, axis=0)
        triangles = np.append(triangles, triangles_i, axis=0)
    if len(vertices) == 0:
        return None
    vertices = o3d.utility.Vector3dVector(vertices)
    triangles = o3d.utility.Vector3iVector(triangles)
    mesh = o3d.geometry.TriangleMesh(vertices, triangles)
    # mesh.compute_vertex_normals(normalized=True)
    # mesh.compute_triangle_normals(normalized=True)
    return mesh


def mesh2pcd(mesh, sample_density, num_points=None):
    pcd_tmp = mesh.sample_points_uniformly(number_of_points=sample_density)
    points = np.asarray(pcd_tmp.points)
    normals = np.asarray(pcd_tmp.normals)

    pcd = o3d.geometry.PointCloud()
    if num_points:
        idx = np.arange(points.shape[0])
        np.random.shuffle(idx)
        idx = idx[:num_points]
        points = points[idx]
        normals = normals[idx]
    # print(vertices.shape, normals.shape)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


# ---------------------------------------------------------------------------- #
# Build from numpy
# ---------------------------------------------------------------------------- #


def np2mesh(vertices, triangles, colors=None, vertex_normals=None, triangle_normals=None):
    """Convert numpy array to open3d PointCloud."""
    # print(vertices, triangles)(
    # print(vertices.dtype, vertices.shape)
    vertices = o3d.utility.Vector3dVector(vertices.copy())
    triangles = o3d.utility.Vector3iVector(triangles.copy())
    # print(vertices, triangles)
    # exit(0)
    mesh = o3d.geometry.TriangleMesh(vertices, triangles)
    if colors is not None:
        colors = colors.copy()
        if colors.ndim == 2:
            assert len(colors) == len(vertices)
        elif colors.ndim == 1:
            colors = np.tile(colors, (len(vertices), 1))
        else:
            raise RuntimeError(colors.shape)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    if vertex_normals is not None:
        assert len(triangles) == len(vertex_normals)
        mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
    # else:
    # mesh.compute_vertex_normals(normalized=True)

    if triangle_normals is not None:
        assert len(triangles) == len(triangle_normals)
        mesh.triangle_normals = o3d.utility.Vector3dVector(triangle_normals)
    # else:
    # mesh.compute_triangle_normals(normalized=True)
    return mesh


def np2pcd(points, colors=None, normals=None):
    """Convert numpy array to open3d PointCloud."""
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.copy())
    if colors is not None:
        colors = np.array(colors)
        if colors.ndim == 2:
            assert len(colors) == len(points)
        elif colors.ndim == 1:
            colors = np.tile(colors, (len(points), 1))
        else:
            raise RuntimeError(colors.shape)
        pc.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        assert len(points) == len(normals)
        pc.normals = o3d.utility.Vector3dVector(normals)
    return pc


def create_aabb(bbox, color=(0, 1, 0)):
    """Draw an axis-aligned bounding box."""
    assert len(bbox) == 6, f"The format of bbox should be xyzwlh, but received {len(bbox)}."
    bbox = np.asarray(bbox)
    abb = o3d.geometry.AxisAlignedBoundingBox(bbox[0:3] - bbox[3:6] * 0.5, bbox[0:3] + bbox[3:6] * 0.5)
    abb.color = color
    return abb


def create_obb(bbox, R=np.eye(3), color=(0, 1, 0)):
    """Draw an oriented bounding box."""
    assert len(bbox) == 6, f"The format of bbox should be xyzwlh, but received {len(bbox)}."
    obb = o3d.geometry.OrientedBoundingBox(bbox[0:3], R, bbox[3:6])
    obb.color = color
    return obb


# ---------------------------------------------------------------------------- #
# Computation
# ---------------------------------------------------------------------------- #
def compute_pcd_normals(points, search_param=None, camera_location=(0.0, 0.0, 0.0)):
    """Compute normals."""
    pcd = np2pcd(points)
    if search_param is None:
        pcd.estimate_normals()
    else:
        pcd.estimate_normals(search_param=search_param)
    pcd.orient_normals_towards_camera_location(camera_location)
    normals = np.array(pcd.normals)
    return normals


def pcd_voxel_down_sample(points, voxel_size, min_bound=(-5.0, -5.0, -5.0), max_bound=(5.0, 5.0, 5.0)):
    """Downsample the point cloud and return sample indices."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downsample_pcd, mapping, index_buckets = pcd.voxel_down_sample_and_trace(voxel_size, np.array(min_bound)[:, None], np.array(max_bound)[:, None])
    sample_indices = [int(x[0]) for x in index_buckets]
    return sample_indices


def create_aabb_from_pcd(pcd, color=(0, 1, 0)):
    if hasattr(pcd, "points"):
        pcd = pcd.points
    elif isinstance(pcd, np.ndarray):
        pcd = o3d.utility.Vector3dVector(pcd)
    abb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(pcd)
    abb.color = color
    return abb


def create_obb_from_pcd(pcd, color=(0, 1, 0)):
    if hasattr(pcd, "points"):
        pcd = pcd.points
    elif isinstance(pcd, np.ndarray):
        pcd = o3d.utility.Vector3dVector(pcd)
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(pcd)
    obb.color = color
    return obb


def create_aabb_from_mesh(mesh, color=(0, 1, 0)):
    return create_aabb_from_pcd(mesh.vertices, color)


def create_obb_from_mesh(mesh, color=(0, 1, 0)):
    return create_obb_from_pcd(mesh.vertices, color)
