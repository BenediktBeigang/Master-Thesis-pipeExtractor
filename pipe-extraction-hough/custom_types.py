from typing import Annotated, Tuple, Union
from numpy.typing import NDArray
import numpy as np

# Point
Point2D = Annotated[NDArray[np.floating], "Shape: (2,)"]
Point3D = Annotated[NDArray[np.floating], "Shape: (3,)"]

# Array
Point2DArray = Annotated[NDArray[np.floating], "Shape: (2, N)"]
Point3DArray = Annotated[NDArray[np.floating], "Shape: (3, N)"]


def Point3DArray_One(point: Point3D) -> Point3DArray:
    return point.reshape((1, 3))


# Bucket
ListOfPoint2DArrays = Annotated[list[Point2DArray], "Each element shape: (M_i, 2)"]
ListOfPoint3DArrays = Annotated[list[Point3DArray], "Each element shape: (M_i, 3)"]

# Segment
Segment2D = Annotated[NDArray[np.floating], "Shape: (2, 2)"]
Segment3D = Annotated[NDArray[np.floating], "Shape: (2, 3)"]

# Segment-Array
Segment2DArray = Annotated[NDArray[np.floating], "Shape: (N, 2, 2)"]
Segment3DArray = Annotated[NDArray[np.floating], "Shape: (N, 2, 3)"]


def Segment3D_Create(start: Point3D, end: Point3D) -> Segment3D:
    return np.vstack([start, end])


def Segment2DArray_Empty():
    return np.empty((0, 2, 2), dtype=np.float64)


def Segment3DArray_Empty():
    return np.empty((0, 2, 3), dtype=np.float64)


def Segment3DArray_One(segment: Segment3D) -> Segment3DArray:
    return segment.reshape((1, 2, 3))
