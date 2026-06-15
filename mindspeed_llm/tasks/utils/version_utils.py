# coding=utf-8
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
#
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


def transformers_version():
    import transformers

    version_idents = transformers.__version__.split('.')
    major_version = int(version_idents[0])
    minor_version = int(version_idents[1])
    return major_version, minor_version


def verify_transformers_version(min_version=None, max_version=None):
    current = transformers_version()
    if min_version and current < min_version:
        raise ValueError(
            f"the version transformers should greater or equal "
            f"{min_version[0]}.{min_version[1]}, "
            f"but got {current[0]}.{current[1]}"
        )
    if max_version and current > max_version:
        raise ValueError(
            f"the version transformers should be less than or equal to "
            f"{max_version[0]}.{max_version[1]}, "
            f"but got {current[0]}.{current[1]}"
        )
