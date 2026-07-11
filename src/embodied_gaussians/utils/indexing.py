import numpy as np


def scalar_index(value, upper_bound: int | None = None) -> int:
    index = int(np.asarray(value).item())
    if upper_bound is not None:
        index = max(0, min(index, upper_bound - 1))
    return index
