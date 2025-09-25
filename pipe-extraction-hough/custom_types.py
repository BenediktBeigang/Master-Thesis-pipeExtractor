from typing import Annotated, Tuple, Union
from numpy.typing import NDArray
import numpy as np

# Point
Point2D = Annotated[NDArray[np.floating], "Shape: (2,)"]
Point3D = Annotated[NDArray[np.floating], "Shape: (3,)"]

# Array
Point2DArray = Annotated[NDArray[np.floating], "Shape: (2, N)"]
Point3DArray = Annotated[NDArray[np.floating], "Shape: (3, N)"]

# Bucket
ListOfPoint2DArrays = Annotated[list[Point2DArray], "Each element shape: (M_i, 2)"]
ListOfPoint3DArrays = Annotated[list[Point3DArray], "Each element shape: (M_i, 3)"]

# Segment
Segment2D = Annotated[NDArray[np.floating], "Shape: (2, 2)"]
Segment3D = Annotated[NDArray[np.floating], "Shape: (2, 3)"]

# Segment-Array
Segment2DArray = Annotated[NDArray[np.floating], "Shape: (N, 2, 2)"]
Segment3DArray = Annotated[NDArray[np.floating], "Shape: (N, 2, 3)"]
