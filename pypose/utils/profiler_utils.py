from torch.profiler import record_function
import functools


def profile_function(label):
    """Decorator for profiling a function with torch.profiler.record_function"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with record_function(label):  # Label function execution
                return func(*args, **kwargs)

        return wrapper

    return decorator
