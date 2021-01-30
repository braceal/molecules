import torch.nn
import wandb
from pathlib import Path
from typing import Optional
from molecules.config import ModelBaseConfig


def setup_wandb(
    cfg: ModelBaseConfig,
    model: torch.nn.Module,
    model_path: Path,
) -> Optional[wandb.config]:
    if cfg.wandb_project_name is not None:
        wandb.init(
            project=cfg.wandb_project_name,
            entity=cfg.wandb_entity_name,
            name=cfg.model_tag,
            id=cfg.model_tag,
            dir=str(model_path),
            config=cfg.dict(),
            resume=False,
        )
        wandb.watch(model)
        return wandb.config
