import json
import yaml
from pydantic import BaseSettings as _BaseSettings
from pathlib import Path
from typing import TypeVar, Type, Union, Optional, Dict, Any

_T = TypeVar("_T")


class BaseSettings(_BaseSettings):
    def dump_yaml(self, cfg_path):
        with open(cfg_path, mode="w") as fp:
            yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: Type[_T], filename: Union[str, Path]) -> _T:
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)


class ModelBaseConfig(BaseSettings):
    # Project name for wandb logging
    wandb_project_name: Optional[str] = None
    # Team name for wandb logging
    wandb_entity_name: Optional[str] = None
    # Model tag for wandb labeling
    model_tag: Optional[str] = None


class OptimizerConfig(BaseSettings):
    class Config:
        extra = "allow"

    name: str = "Adam"
    hparams: Dict[str, Any] = {}
