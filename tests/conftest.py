import sys
from pathlib import Path

# Load torch's native DLLs FIRST: importing pyarrow before torch intermittently
# segfaults on Windows (DLL conflict); torch-first is the known-stable order.
import torch  # noqa: F401  (import order matters, see above)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
