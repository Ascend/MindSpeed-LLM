
import os
import time
import random
from functools import wraps


def auto_coverage(func):
    """
    Decide whether to collect coverage based on the START_COVERAGE environment variable.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check the environment variable.
        if os.environ.get('START_COVERAGE', '').lower() != 'true':
            return func(*args, **kwargs)
        
        import coverage
        cov = coverage.Coverage(data_suffix=f"usecase-{time.time_ns()}_{random.randint(0, 100)}")
        # Collect coverage.
        cov.start()
        try:
            return func(*args, **kwargs)
        finally:
            # Stop coverage.
            cov.stop()
            # Save coverage data.
            cov.save()
    
    return wrapper