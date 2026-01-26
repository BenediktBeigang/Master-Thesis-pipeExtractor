from typing import Annotated, List
from numpy.typing import NDArray
import numpy as np

# Point
Point2D = Annotated[NDArray[np.float64], "Shape: (2,)"]
Point3D = Annotated[NDArray[np.float64], "Shape: (3,)"]

# Array
Point2DArray = Annotated[NDArray[np.float64], "Shape: (N, 2)"]
Point3DArray = Annotated[NDArray[np.float64], "Shape: (N, 3)"]


def Point3DArray_One(point: Point3D) -> Point3DArray:
    return point.reshape((1, 3))


# List of buckets / List of chains
ListOfPoint2DArrays = Annotated[List[Point2DArray], "Each element shape: (M_i, 2)"]
ListOfPoint3DArrays = Annotated[List[Point3DArray], "Each element shape: (M_i, 3)"]

# Segment
Segment2D = Annotated[NDArray[np.float64], "Shape: (2, 2)"]
Segment3D = Annotated[NDArray[np.float64], "Shape: (2, 3)"]

# Segment-Array
Segment2DArray = Annotated[NDArray[np.float64], "Shape: (N, 2, 2)"]
Segment3DArray = Annotated[NDArray[np.float64], "Shape: (N, 2, 3)"]


def Segment3D_Create(start: Point3D, end: Point3D) -> Segment3D:
    return np.stack((start, end), axis=0)


def Segment2DArray_Empty():
    return np.empty((0, 2, 2), dtype=np.float64)


def Segment3DArray_Empty():
    return np.empty((0, 2, 3), dtype=np.float64)


def Segment3DArray_One(segment: Segment3D) -> Segment3DArray:
    return segment.reshape((1, 2, 3))


PipeComponent = Annotated[tuple[Point3D, Point3D, Point3D], "bbox_min, bbox_max, mean"]
PipeComponentArray = Annotated[List[PipeComponent], "List of PipeComponents"]
