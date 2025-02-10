from typing import TypeAlias, List, Tuple, Dict
import numpy as np
from numpy.typing import NDArray

# Represents grayscale image as a 2D array of float32
Image: TypeAlias = NDArray[np.float32]

# Represents gradients computed from voronoi borders as a 2D NumPy array of float32
GradientImage: TypeAlias = NDArray[np.float32]

# Represents vertices as a 2D NumPy array of float64, each vertex being [x, y]
Vertices: TypeAlias = NDArray[np.float64]

# Represents a region defined by indices of vertices in the Vertices array
IndexedRegion: TypeAlias = List[int]

# Represents a sorted edge as a tuple of two integers referencing indices in Vertices
Edge: TypeAlias = Tuple[int]

# Represents a point as a tuple of two integers (x, y)
Point: TypeAlias = Tuple[int, int]

# Represents a NumPy random number generator
RandomState: TypeAlias = np.random.RandomState

# Represents a binary mask for Voronoi borders as a 2D NumPy array of uint8
BorderMask: TypeAlias = NDArray[np.uint8]

# Represents a 1D NumPy array of integers for x or y coordinates
Coords: TypeAlias = NDArray[int]

# A ridge is represented as a tuple of (adjacent_point_index, ridge_vertex_index1, ridge_vertex_index2)
VoronoiRidge: TypeAlias = Tuple[int, int, int]

# Dictionary where keys are ridge point indices and values are lists of ridge tuples
AllRidges: TypeAlias = Dict[int, List[VoronoiRidge]]

# Consists of two float coordinates stored in a NumPy array
FloatPoint: TypeAlias = NDArray[np.float64]

# Three ints for grayscale color
# Todo Should change to float
Color: TypeAlias = List[int]

# Binary mask with precipitates
Mask: TypeAlias = NDArray[np.float32]

# NDArray of shape (num_spline_points, 1, 2) used by cv2 to fill contours
Contour: TypeAlias = NDArray[np.int32]

#
