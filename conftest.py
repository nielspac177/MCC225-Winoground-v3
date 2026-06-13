import sys
from pathlib import Path

# Asegura que la raíz del repo esté en sys.path para `import src.*`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
