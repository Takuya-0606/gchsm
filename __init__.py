# Copyright 2025 The GCHSM Developers. All Rights Reserved.
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
#
# Author: Takuya HASHIMOTO
#

# GCHSM 1.0 driver package
from __future__ import annotations
from importlib import metadata
from typing import List, Optional
from .main import main as _main
__all__ = ["__version__", "run_cli"]
try:
    __version__ = metadata.version("gchsm")
except metadata.PackageNotFoundError:  # pragma: no cover - local source tree
    __version__ = "1.0"

def run_cli(argv: Optional[List[str]] = None) -> int:
    """Run the command-line interface and return an exit status."""
    return _main(argv)
