from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from google.cloud.storage.blob import Blob
from pydantic import BaseModel, validator

from .lib.gcp_storage import StorageBucket
from .lib.types import InstrumentId, PartBlob


def is_blob_a_part(blob: Blob) -> PartBlob | Literal[False]:
    blob_path = Path(blob.name)
    if blob_path.suffix != ".pdf":
        return False
    if InstrumentId.value_exists(blob_path.stem):
        return PartBlob(instrument_id=blob_path.stem, path=blob_path, blob=blob)
    if InstrumentId.value_exists(blob_path.parent.stem):
        return PartBlob(instrument_id=blob_path.parent.stem, path=blob_path, blob=blob)
    return False


class Input(BaseModel):
    storage_path: Path

    @validator("storage_path")
    def validate_blob(cls: Input, value: Path) -> Path:
        # sourcery skip: instance-method-first-arg-name
        cls.get_part_blob(value)
        return value

    @staticmethod
    def get_part_blob(storage_path: Path = None) -> PartBlob:
        storage_path = storage_path
        if not (blob := StorageBucket.get_blob(storage_path.as_posix())):
            raise ValueError(
                f"No file can be found in storage in the path {storage_path}"
            )
        if not (part_blob := is_blob_a_part(blob)):
            raise ValueError(f"File in the path {storage_path} is not a part")
        return part_blob

    @property
    def part_blob(self) -> PartBlob:
        return self.get_part_blob(self.storage_path)


class InputWithInstrumentId(BaseModel):
    instrument_id: InstrumentId
    storage_path: Path

    @validator("storage_path")
    def validate_blob(
        cls: InputWithInstrumentId, value: Path, values: dict[str, Any]
    ) -> Path:
        # sourcery skip: instance-method-first-arg-name
        cls.get_part_blob(value, values["instrument_id"])
        return value

    @staticmethod
    def get_part_blob(
        storage_path: Path = None, instrument_id: InstrumentId = None
    ) -> PartBlob:
        storage_path = storage_path
        if not (blob := StorageBucket.get_blob(storage_path.as_posix())):
            raise ValueError(
                f"No file can be found in storage in the path {storage_path}"
            )
        return PartBlob(instrument_id=instrument_id, path=storage_path, blob=blob)

    @property
    def part_blob(self) -> PartBlob:
        return self.get_part_blob(self.storage_path)
