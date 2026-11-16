# Copyright 2024 Huawei Technologies Co., Ltd.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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