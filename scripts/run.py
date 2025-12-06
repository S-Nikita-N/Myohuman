import logging
import sys
import torch
import wandb
import numpy as np
import torch.multiprocessing as mp

from typing import List
from hydra import main as hydra_main
from omegaconf import OmegaConf, DictConfig

from myohuman.agents.agent_im import AgentIM


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def setup_reproducibility(cfg: DictConfig):
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = (
        torch.device("cuda", index=0)
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )
    logging.info("Using device: %s", device)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return dtype, device


def init_wandb(cfg: DictConfig) -> None:
    group = cfg.get("group", cfg.learning.agent_name)
    wandb.init(
        project=cfg.project,
        group=group,
        resume=cfg.resume_str is not None,
        id=cfg.resume_str,
        notes=cfg.notes,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
    )
    wandb.run.name = cfg.exp_name
    wandb.log({"config": OmegaConf.to_container(cfg, resolve=True)})


def run_train(cfg: DictConfig) -> None:
    dtype, device = setup_reproducibility(cfg)
    agent = AgentIM(
        cfg=cfg,
        dtype=dtype,
        device=device,
        training=True,
        checkpoint_epoch=cfg.epoch,
    )
    agent.optimize_policy()
    logging.info("Training done.")


def run_eval(cfg: DictConfig) -> None:
    dtype, device = setup_reproducibility(cfg)
    agent = AgentIM(
        cfg=cfg,
        dtype=dtype,
        device=device,
        training=False,
        checkpoint_epoch=cfg.epoch,
    )
    agent.eval_policy(epoch=cfg.epoch)
    logging.info("Evaluation done.")


@hydra_main(config_path="../cfg", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    mp.set_start_method("fork", force=True)

    if cfg.get("im_eval", False):
        run_eval(cfg)
    else:
        if not cfg.no_log and not cfg.run.test:
            init_wandb(cfg)
        run_train(cfg)


if __name__ == "__main__":
    main()
