"""General utilities."""

from __future__ import annotations

import time
from functools import wraps


def timeit(func):
    """Lightweight timing decorator for notebooks/scripts."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"La funcion demoro {end - start:.2f} segundos")
        return result

    return wrapped
