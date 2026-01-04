from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_raw: Path = project_root / "data" / "raw" / "heart.csv"
    model_dir: Path = project_root / "models"
    model_path: Path = model_dir / "model.pkl"
    metadata_path: Path = model_dir / "metadata.json"
    artifacts_dir: Path = project_root / "artifacts"

DEFAULT_SEED = 42
