import argparse, os, sys, datetime, glob, importlib, csv
import time
import torch
import torchvision
import numpy as np
import pytorch_lightning as pl
from data.datasets import StreetSatTrain, StreetSatTest, StreetSatVal

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image
from ldm.modules.utils import instantiate_from_config

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.loggers import TensorBoardLogger

def get_parser(**parser_kwargs):

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="log directory",
    )

    return parser

class StreetSatDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, val=None, test=None,
                 num_workers=None, use_worker_init_fn=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn

        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if val is not None:
            self.dataset_configs["val"] = val
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader

    def prepare_data(self):
        for cfg in self.dataset_configs.values():
            instantiate_from_config(cfg)

    def setup(self, stage=None):
        self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)
    
    def _val_dataloader(self):
        return DataLoader(self.datasets["val"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)
    
    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)


if __name__ == "__main__":
    parser = get_parser()

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    sys.path.append(os.getcwd())

    args, unk = parser.parse_known_args()

    if args.name and args.resume:
        raise ValueError("use --name for initialization, --resume for resuming intialized model")
    
    if args.resume:
        logdir = args.resume.rstrip("/")
        ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        args.resume_from_checkpoint = ckpt
    else:
        cfg_fname = os.path.splitext(os.path.split(args.base[0])[-1])[0]
        nowname = now + "_" + cfg_fname
        logdir = os.path.join(args.logdir, nowname)
        os.makedirs(logdir, exist_ok=True)
    
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)

    cfg = OmegaConf.load(args.base[0])
    for cfg_path in args.base[1:]:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg_path))
    
    data_module = instantiate_from_config(cfg.data)
    data_module.prepare_data()
    data_module.setup()
    model = instantiate_from_config(cfg.model)
    
    lightning_cfg = cfg.lightning

    # callback management

    if "callbacks" in lightning_cfg:
        callbacks_cfg = lightning_cfg.callbacks
    else:
        callbacks_cfg = OmegaConf.create()

    metrics_over_trainsteps_ckpt_dict = {
        'metrics_over_trainsteps_checkpoint':
            {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                'params': {
                    "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                    "filename": "{epoch:06}-{step:09}",
                    "verbose": True,
                    'save_top_k': 3,
                    'every_n_train_steps': 10000,
                    'save_weights_only': True
                }
            }
    }

    callbacks_cfg.update(metrics_over_trainsteps_ckpt_dict)

    callbacks = [instantiate_from_config(cb) for cb in cfg.lightning.callbacks.values()]

    # trainer configurations

    trainer_cfg = lightning_cfg.get("trainer", OmegaConf.create())

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_gpus}")
        trainer_cfg["gpus"] = num_gpus
        trainer_cfg["accelerator"] = "ddp"
    else:
        print("No GPUs found, using CPU.")
        trainer_cfg["gpus"] = 0

    logger = TensorBoardLogger(save_dir=logdir, name='StreetSat')

    trainer = Trainer(
        logger=logger,
        resume_from_checkpoint=args.resume_from_checkpoint if args.resume else None,
        max_epochs=trainer_cfg.get("max_epochs", 100),
        callbacks=callbacks,
        **trainer_cfg
    )

    model = instantiate_from_config(cfg.model)
    
    bs, base_lr = cfg.data.params.batch_size, cfg.model.base_learning_rate

    if 'accumulate_grad_batches' in lightning_cfg.trainer:
        accumulate_grad_batches = lightning_cfg.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1

    model.learning_rate = accumulate_grad_batches * num_gpus * bs * base_lr
    
    if args.train:
        trainer.fit(model, data_module)