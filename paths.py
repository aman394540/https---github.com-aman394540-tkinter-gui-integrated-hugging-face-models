from pathlib import Path

# Base project directory (this file is in app/utils)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)
