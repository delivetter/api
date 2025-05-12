import os

from .models import ConfigType


def import_config() -> ConfigType:
    return ConfigType.model_validate_json(os.environ["CONFIG"])
