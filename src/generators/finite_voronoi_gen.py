from typing import Tuple, List, Optional, Dict

import numpy as np
from scipy.spatial import Voronoi
from type_definitions import IndexedRegion, Vertices, AllRidges, FloatPoint, Edge


def _build_all_ridges(vor: Voronoi) -> AllRidges:
    """ Build a dictionary mapping each ridge point index to its corresponding ridge segments. """
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    return all_ridges


def _is_region_finite(region: IndexedRegion) -> bool:
    """ Determine if a Voronoi region is finite. """
    return all(v >= 0 for v in region)


def _get_finite_vertices(region: IndexedRegion) -> IndexedRegion:
    """ Extract finite vertex indices from a Voronoi region. """
    return [v for v in region if v >= 0]


def _sort_region_counterclockwise(new_vertices: Vertices, region: IndexedRegion) -> IndexedRegion:
    """ Sort the vertices of the given region counterclockwise for proper polygon ordering. """
    vs = new_vertices[region]
    c = vs.mean(axis=0)
    angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
    return np.array(region)[np.argsort(angles)].tolist()


def _compute_far_point(p1: int, p2: int, v2: int, vor: Voronoi, radius: float) -> FloatPoint:
    """ Compute the 'far point' for an infinite Voronoi ridge. """
    tangent = vor.points[p2] - vor.points[p1]
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])

    midpoint = vor.points[[p1, p2]].mean(axis=0)
    center = vor.points.mean(axis=0)
    direction = np.sign(np.dot(midpoint - center, normal)) * normal
    far_point = vor.vertices[v2] + direction * radius

    return far_point


def _make_region_finite(p1: int, region: IndexedRegion, vor: Voronoi, radius: float, new_vertices: Vertices,
                        added_points: Dict[Edge, int]) -> Tuple[IndexedRegion, Vertices]:
    """ For a region that has infinite edges, extend these edges to 'far points' making the region finite. """
    new_region = _get_finite_vertices(region)
    all_ridges = _build_all_ridges(vor)

    # Examine each ridge for potential infinite edges
    for p2, v1, v2 in all_ridges[p1]:
        if v2 < 0:
            v1, v2 = v2, v1
        if v1 >= 0:
            # finite ridge: already in the region
            continue

        # Check if the far point was already computed for this edge
        sorted_edge = tuple(sorted((p1, p2)))
        if sorted_edge in added_points:
            point_index = added_points[sorted_edge]
        else:
            far_point = _compute_far_point(p1, p2, v2, vor, radius)
            new_vertices = np.vstack([new_vertices, far_point])
            point_index = len(new_vertices) - 1
            added_points[sorted_edge] = point_index

        new_region.append(point_index)

    sorted_region = _sort_region_counterclockwise(new_vertices, new_region)
    return sorted_region, new_vertices


def get_finite_voronoi(vor: Voronoi, radius: Optional[float] = None) -> Tuple[List[IndexedRegion], Vertices]:
    """
    Reconstruct infinite Voronoi regions into finite polygons.

    Parameters
    ----------
    vor : Voronoi
        Input Voronoi diagram (2D).
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of lists
        Indices of vertices in each finite Voronoi region.
    vertices : ndarray
        Coordinates of the finite Voronoi vertices.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    if radius is None:
        radius = 2

    added_points = {}
    new_regions = []
    new_vertices = vor.vertices.copy()

    # Reconstruct regions
    for p1, region_index in enumerate(vor.point_region):
        region = vor.regions[region_index]
        if _is_region_finite(region):
            new_regions.append(region)
        else:
            new_region, new_vertices = _make_region_finite(p1, region, vor, radius, new_vertices, added_points)
            new_regions.append(new_region)

    return new_regions, new_vertices
