"""JSON file-based storage implementation."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from ..models.module import Module
from ..models.pipeline import Pipeline
from ..utils.logger import get_logger

logger = get_logger(__name__)


class JSONStorage:
    """JSON file-based storage for modules and pipelines."""

    def __init__(self, storage_dir: str = "data"):
        """Initialize JSON storage.

        Args:
            storage_dir: Directory to store JSON files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        self.modules_file = self.storage_dir / "modules.json"
        self.pipelines_file = self.storage_dir / "pipelines.json"

        # Initialize files if they don't exist
        self._ensure_files_exist()

    def _ensure_files_exist(self) -> None:
        """Ensure storage files exist with empty data."""
        if not self.modules_file.exists():
            self._write_json(self.modules_file, {})
        if not self.pipelines_file.exists():
            self._write_json(self.pipelines_file, {})

    def _read_json(self, file_path: Path) -> Dict[str, Any]:
        """Read JSON data from file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return {}

    def _write_json(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Write JSON data to file."""
        try:
            # Create backup
            if file_path.exists():
                backup_path = file_path.with_suffix(f".backup.{int(datetime.now().timestamp())}")
                file_path.rename(backup_path)

                # Keep only latest 5 backups
                backups = sorted(file_path.parent.glob(f"{file_path.stem}.backup.*"))
                for old_backup in backups[:-5]:
                    old_backup.unlink()

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=self._json_serializer)

        except Exception as e:
            logger.error(f"Failed to write {file_path}: {e}")
            raise

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for complex objects."""
        if isinstance(obj, UUID):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, "dict"):
            return obj.dict()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # Module operations
    def save_module(self, module: Module) -> None:
        """Save a module to storage."""
        modules_data = self._read_json(self.modules_file)
        modules_data[str(module.id)] = module.dict()
        self._write_json(self.modules_file, modules_data)
        logger.info(f"Saved module {module.id} to storage")

    def load_module(self, module_id: UUID) -> Optional[Module]:
        """Load a module from storage."""
        modules_data = self._read_json(self.modules_file)
        module_data = modules_data.get(str(module_id))

        if module_data:
            try:
                return Module(**module_data)
            except Exception as e:
                logger.error(f"Failed to deserialize module {module_id}: {e}")
                return None
        return None

    def load_all_modules(self) -> List[Module]:
        """Load all modules from storage."""
        modules_data = self._read_json(self.modules_file)
        modules = []

        for module_id, module_data in modules_data.items():
            try:
                module = Module(**module_data)
                modules.append(module)
            except Exception as e:
                logger.error(f"Failed to deserialize module {module_id}: {e}")
                continue

        return modules

    def delete_module(self, module_id: UUID) -> bool:
        """Delete a module from storage."""
        modules_data = self._read_json(self.modules_file)
        if str(module_id) in modules_data:
            del modules_data[str(module_id)]
            self._write_json(self.modules_file, modules_data)
            logger.info(f"Deleted module {module_id} from storage")
            return True
        return False

    # Pipeline operations
    def save_pipeline(self, pipeline: Pipeline) -> None:
        """Save a pipeline to storage."""
        pipelines_data = self._read_json(self.pipelines_file)
        pipelines_data[str(pipeline.id)] = pipeline.dict()
        self._write_json(self.pipelines_file, pipelines_data)
        logger.info(f"Saved pipeline {pipeline.id} to storage")

    def load_pipeline(self, pipeline_id: UUID) -> Optional[Pipeline]:
        """Load a pipeline from storage."""
        pipelines_data = self._read_json(self.pipelines_file)
        pipeline_data = pipelines_data.get(str(pipeline_id))

        if pipeline_data:
            try:
                return Pipeline(**pipeline_data)
            except Exception as e:
                logger.error(f"Failed to deserialize pipeline {pipeline_id}: {e}")
                return None
        return None

    def load_all_pipelines(self) -> List[Pipeline]:
        """Load all pipelines from storage."""
        pipelines_data = self._read_json(self.pipelines_file)
        pipelines = []

        for pipeline_id, pipeline_data in pipelines_data.items():
            try:
                pipeline = Pipeline(**pipeline_data)
                pipelines.append(pipeline)
            except Exception as e:
                logger.error(f"Failed to deserialize pipeline {pipeline_id}: {e}")
                continue

        return pipelines

    def delete_pipeline(self, pipeline_id: UUID) -> bool:
        """Delete a pipeline from storage."""
        pipelines_data = self._read_json(self.pipelines_file)
        if str(pipeline_id) in pipelines_data:
            del pipelines_data[str(pipeline_id)]
            self._write_json(self.pipelines_file, pipelines_data)
            logger.info(f"Deleted pipeline {pipeline_id} from storage")
            return True
        return False

    # Utility methods
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        modules_data = self._read_json(self.modules_file)
        pipelines_data = self._read_json(self.pipelines_file)

        return {
            "modules_count": len(modules_data),
            "pipelines_count": len(pipelines_data),
            "storage_dir": str(self.storage_dir),
            "modules_file_size": (
                self.modules_file.stat().st_size if self.modules_file.exists() else 0
            ),
            "pipelines_file_size": (
                self.pipelines_file.stat().st_size if self.pipelines_file.exists() else 0
            ),
        }

    def backup_data(self, backup_name: Optional[str] = None) -> str:
        """Create a backup of all data."""
        if backup_name is None:
            backup_name = f"backup_{int(datetime.now().timestamp())}"

        backup_dir = self.storage_dir / "backups" / backup_name
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Copy data files
        if self.modules_file.exists():
            import shutil

            shutil.copy2(self.modules_file, backup_dir / "modules.json")

        if self.pipelines_file.exists():
            import shutil

            shutil.copy2(self.pipelines_file, backup_dir / "pipelines.json")

        logger.info(f"Created backup: {backup_dir}")
        return str(backup_dir)
