# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import threading
from typing import Optional

_lock = threading.Lock()
_model_weight_bytes: Optional[int] = None


def set_model_weight_bytes(model_weight_bytes: int) -> None:
    """Store model weight memory size (bytes) for reporting."""
    if model_weight_bytes < 0:
        raise ValueError("model_weight_bytes must be non-negative")
    with _lock:
        global _model_weight_bytes
        _model_weight_bytes = int(model_weight_bytes)


def get_model_weight_bytes() -> Optional[int]:
    with _lock:
        return _model_weight_bytes

