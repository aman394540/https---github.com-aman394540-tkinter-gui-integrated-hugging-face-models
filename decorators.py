import time
import functools

def timeit(fn):
    """Decorator: measure runtime of a function and attach `last_runtime_s` attribute to the instance."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        start = time.time()
        try:
            return fn(self, *args, **kwargs)
        finally:
            self.last_runtime_s = time.time() - start
    return wrapper

def ensure_initialized(fn):
    """Decorator: ensure the model has been initialized before using it."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "_initialized", False):
            raise RuntimeError("Model is not initialized. Call `load()` first.")
        return fn(self, *args, **kwargs)
    return wrapper
