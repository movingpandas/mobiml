from typing import Tuple, Union, List
import numpy as np

class XY(Tuple):
    def __init__(x: np.ndarray, y: np.ndarray):
        super.__init__(x, y)

class Dataset(Tuple):
    def __init__(a: XY, b: XY):
        super.__init__(a, b)

class LogRegParams(Tuple):
    def __init__(a: XY, b:Tuple[np.ndarray]):
        super.__init__(a, b)

class XYList(List):
    def __init__(a: XY):
        super.__init__(a)

