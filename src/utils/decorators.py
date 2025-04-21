import os
import sys
from functools import wraps


def suppress_prints(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Redirect stdout to devnull
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            return func(*args, **kwargs)
        finally:
            # Restore original stdout
            sys.stdout.close()
            sys.stdout = original_stdout

    return wrapper
