import os
from typing import TYPE_CHECKING


# You can ignore this conditional. It just defaults your IDE type hints to numpy so they are not broken by the conditional import.
if TYPE_CHECKING:
    import numpy as xp
else:
    if os.getenv("USE_GPU") == "1":
        try:
            import cupy as xp
            print("Backend: CuPy (GPU)")
        except ImportError:
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            import numpy as xp
            print("Backend: NumPy (CPU - CuPy not found). Run `uv sync --extra cuda` to install CuPy.")
    else:
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        import numpy as xp
        print("Backend: NumPy (CPU)")

