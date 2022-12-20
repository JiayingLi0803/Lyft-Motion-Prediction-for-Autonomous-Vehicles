#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Predict multi-mode with confidence: As written in evaluation metric page, we can predict 3 modes of motion trajectory.
#Training loss with competition evaluation metric
#Use Training abstraction library pytorch-ignite and pytorch-pfn-extras.


# In[ ]:


#https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/evaluation/csv_utils.py#L140


# In[ ]:


#https://github.com/lyft/l5kit/blob/master/competition.md


# In[1]:


import gc
import os
from pathlib import Path
import random
import sys

from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy as sp


import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

# --- plotly ---
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
pio.templates.default = "plotly_dark"

# --- models ---
from sklearn import preprocessing
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# --- setup ---
pd.set_option('max_columns', 50)


# In[2]:


import torch
from pathlib import Path

import pytorch_pfn_extras as ppe
from math import ceil
from pytorch_pfn_extras.training import IgniteExtensionsManager
from pytorch_pfn_extras.training.triggers import MinValueTrigger

from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import pytorch_pfn_extras.training.extensions as E


# In[3]:


# --- Dataset utils ---
from typing import Callable

from torch.utils.data.dataset import Dataset


class TransformDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Callable):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        batch = self.dataset[index]
        return self.transform(batch)

    def __len__(self):
        return len(self.dataset)


# In[4]:


import zarr

import l5kit
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset

from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR
from l5kit.geometry import transform_points
from tqdm import tqdm
from collections import Counter
from l5kit.data import PERCEPTION_LABELS
from prettytable import PrettyTable
from l5kit.evaluation import write_pred_csv

from matplotlib import animation, rc
from IPython.display import HTML

rc('animation', html='jshtml')
print("l5kit version:", l5kit.__version__)


# In[5]:


#Function
#To define loss function to calculate competition evaluation metric in batch.


# In[6]:


# --- Function utils ---
# Original code from https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/evaluation/metrics.py
import numpy as np

import torch
from torch import Tensor


def pytorch_neg_multi_log_likelihood_batch(
    gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor
) -> Tensor:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (batch_size, future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))), "confidences should sum to 1"
    assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    # print("error", error)
    return torch.mean(error)


def pytorch_neg_multi_log_likelihood_single(
    gt: Tensor, pred: Tensor, avails: Tensor
) -> Tensor:
    """

    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(time)x(2D coords)
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    # pred (bs)x(time)x(2D coords) --> (bs)x(mode=1)x(time)x(2D coords)
    # create confidence (bs)x(mode=1)
    batch_size, future_len, num_coords = pred.shape
    confidences = pred.new_ones((batch_size, 1))
    return pytorch_neg_multi_log_likelihood_batch(gt, pred.unsqueeze(1), confidences, avails)


# In[7]:


#Model
#pytorch model definition. Here model outputs both multi-mode trajectory prediction & confidence of each trajectory.


# In[25]:


# --- Model utils ---
import torch
from torchvision.models import resnet34
from torch import nn
from typing import Dict


class LyftMultiModel(nn.Module):

    def __init__(self, cfg: Dict, num_modes=3):
        super().__init__()

        # TODO: support other than resnet18?
        backbone = resnet34(pretrained=True, progress=True)
        self.backbone = backbone

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )

        # This is 512 for resnet18 and resnet34;
        # And it is 2048 for the other resnets
        backbone_out_features = 512

        # X, Y coords for the future positions (output shape: Bx50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len

        # You can add more layers here.
        self.head = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(in_features=backbone_out_features, out_features=4096),
        )

        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        self.logit = nn.Linear(4096, out_features=self.num_preds + num_modes)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.head(x)
        x = self.logit(x)

        # pred (bs)x(modes)x(time)x(2D coords)
        # confidences (bs)x(modes)
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences


# In[9]:


class LyftMultiRegressor(nn.Module):
    """Single mode prediction"""

    def __init__(self, predictor, lossfun=pytorch_neg_multi_log_likelihood_batch):
        super().__init__()
        self.predictor = predictor
        self.lossfun = lossfun

    def forward(self, image, targets, target_availabilities):
        pred, confidences = self.predictor(image)
        loss = self.lossfun(targets, pred, confidences, target_availabilities)
        metrics = {
            "loss": loss.item(),
            "nll": pytorch_neg_multi_log_likelihood_batch(targets, pred, confidences, target_availabilities).item()
        }
        ppe.reporting.report(metrics, self)
        return loss, metrics


# In[10]:


#Training with Ignite
#I use pytorch-ignite for training abstraction.
#Engine defines the 1 iteration of training update.


# In[11]:


# --- Training utils ---
from ignite.engine import Engine


def create_trainer(model, optimizer, device) -> Engine:
    model.to(device)

    def update_fn(engine, batch):
        model.train()
        optimizer.zero_grad()
        loss, metrics = model(*[elem.to(device) for elem in batch])
        loss.backward()
        optimizer.step()
        return metrics
    trainer = Engine(update_fn)
    return trainer


# In[12]:


# Code referenced from https://github.com/grafi-tt/chaineripy/blob/master/chaineripy/extensions/print_report.py by @grafi-tt
# Modified to work with pytorch_pfn_extras

import os
import sys
from copy import deepcopy

from IPython.core.display import display
from ipywidgets import HTML

from pytorch_pfn_extras.training.extensions.print_report import PrintReport

from pytorch_pfn_extras.training import extension
from pytorch_pfn_extras.training.extensions import log_report     as log_report_module
from pytorch_pfn_extras.training.extensions import util


class PrintReportNotebook(PrintReport):

    """An extension to print the accumulated results.

    This extension uses the log accumulated by a :class:`LogReport` extension
    to print specified entries of the log in a human-readable format.

    Args:
        entries (list of str ot None): List of keys of observations to print.
            If `None` is passed, automatically infer keys from reported dict.
        log_report (str or LogReport): Log report to accumulate the
            observations. This is either the name of a LogReport extensions
            registered to the manager, or a LogReport instance to use
            internally.
        out: Stream to print the bar. Standard output is used by default.

    """

    def __init__(self, entries=None, log_report='LogReport', out=sys.stdout):
        super(PrintReportNotebook, self).__init__(entries=entries, log_report=log_report, out=out)
        self._widget = HTML()

    def initialize(self, trainer):
        display(self._widget)

    @property
    def widget(self):
        return self._widget

    def __call__(self, manager):
        log_report = self.get_log_report(manager)
        df = log_report.to_dataframe()
        if self._infer_entries:
            # --- update entries ---
            self._update_entries(log_report)
        self._widget.value = df[self._entries].to_html(index=False, na_rep='')


# In[13]:


# Code referenced from https://github.com/grafi-tt/chaineripy/blob/master/chaineripy/extensions/progress_bar.py by @grafi-tt
# Modified to work with pytorch_pfn_extras

from pytorch_pfn_extras.training import extension, trigger
import datetime
import time

from IPython.core.display import display
from ipywidgets import FloatProgress, HBox, HTML, VBox


class ProgressBarNotebook(extension.Extension):

    """Trainer extension to print a progress bar and recent training status.
    This extension prints a progress bar at every call. It watches the current
    iteration and epoch to print the bar.
    Args:
        training_length (tuple): Length of whole training. It consists of an
            integer and either ``'epoch'`` or ``'iteration'``. If this value is
            omitted and the stop trigger of the trainer is
            :class:`IntervalTrigger`, this extension uses its attributes to
            determine the length of the training.
        update_interval (int): Number of iterations to skip printing the
            progress bar.
        bar_length (int): Length of the progress bar in characters.
        out: Stream to print the bar. Standard output is used by default.
    """

    def __init__(self, training_length=None, update_interval=100,
                 bar_length=50):
        self._training_length = training_length
        if training_length is not None:
            self._init_status_template()
        self._update_interval = update_interval
        self._recent_timing = []

        self._total_bar = FloatProgress(description='total',
                                        min=0, max=1, value=0,
                                        bar_style='info')
        self._total_html = HTML()
        self._epoch_bar = FloatProgress(description='this epoch',
                                        min=0, max=1, value=0,
                                        bar_style='info')
        self._epoch_html = HTML()
        self._status_html = HTML()

        self._widget = VBox([HBox([self._total_bar, self._total_html]),
                             HBox([self._epoch_bar, self._epoch_html]),
                             self._status_html])

    def initialize(self, manager):
        if self._training_length is None:
            t = manager._stop_trigger
            if not isinstance(t, trigger.IntervalTrigger):
                raise TypeError(
                    'cannot retrieve the training length from %s' % type(t))
            self._training_length = t.period, t.unit
            self._init_status_template()

        updater = manager.updater
        self.update(updater.iteration, updater.epoch_detail)
        display(self._widget)

    def __call__(self, manager):
        length, unit = self._training_length

        updater = manager.updater
        iteration, epoch_detail = updater.iteration, updater.epoch_detail

        if unit == 'iteration':
            is_finished = iteration == length
        else:
            is_finished = epoch_detail == length

        if iteration % self._update_interval == 0 or is_finished:
            self.update(iteration, epoch_detail)

    def finalize(self):
        if self._total_bar.value != 1:
            self._total_bar.bar_style = 'warning'
            self._epoch_bar.bar_style = 'warning'

    @property
    def widget(self):
        return self._widget

    def update(self, iteration, epoch_detail):
        length, unit = self._training_length

        recent_timing = self._recent_timing
        now = time.time()

        recent_timing.append((iteration, epoch_detail, now))

        if unit == 'iteration':
            rate = iteration / length
        else:
            rate = epoch_detail / length
        self._total_bar.value = rate
        self._total_html.value = "{:6.2%}".format(rate)

        epoch_rate = epoch_detail - int(epoch_detail)
        self._epoch_bar.value = epoch_rate
        self._epoch_html.value = "{:6.2%}".format(epoch_rate)

        status = self._status_template.format(iteration=iteration,
                                              epoch=int(epoch_detail))

        if rate == 1:
            self._total_bar.bar_style = 'success'
            self._epoch_bar.bar_style = 'success'

        old_t, old_e, old_sec = recent_timing[0]
        span = now - old_sec
        if span != 0:
            speed_t = (iteration - old_t) / span
            speed_e = (epoch_detail - old_e) / span
        else:
            speed_t = float('inf')
            speed_e = float('inf')

        if unit == 'iteration':
            estimated_time = (length - iteration) / speed_t
        else:
            estimated_time = (length - epoch_detail) / speed_e
        estimate = ('{:10.5g} iters/sec. Estimated time to finish: {}.'
                    .format(speed_t,
                            datetime.timedelta(seconds=estimated_time)))

        self._status_html.value = status + estimate

        if len(recent_timing) > 100:
            del recent_timing[0]

    def _init_status_template(self):
        self._status_template = (
            '{iteration:10} iter, {epoch} epoch / %s %ss<br />' %
            self._training_length)


# In[14]:


# --- Utils ---
import yaml


def save_yaml(filepath, content, width=120):
    with open(filepath, 'w') as f:
        yaml.dump(content, f, width=width)


def load_yaml(filepath):
    with open(filepath, 'r') as f:
        content = yaml.safe_load(f)
    return content


class DotDict(dict):
    """dot.notation access to dictionary attributes

    Refer: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/23689767#23689767
    """  # NOQA

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# In[15]:


#Configs


# In[16]:


# --- Lyft configs ---
cfg = {
    'format_version': 4,
    'model_params': {
        'model_architecture': 'resnet34',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1
    },

    'raster_params': {
        'raster_size': [267,267], # 224*224, 300*300, 350*350, 448*448, you can give a try to set 224*224
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },

    'train_data_loader': {
        'key': 'scenes/train.zarr',
        'batch_size': 32, #  16, 32, you can give a try to set 16
        'shuffle': True,
        'num_workers': 8
    },

    'valid_data_loader': {
        'key': 'scenes/validate.zarr',
        'batch_size': 32, #  16, 32, you can give a try to set 16
        'shuffle': False,
        'num_workers': 8
    },

    'train_params': {
        'max_num_steps': 10000, # 10000, 20000, 30000, 600k, 700k, you can give a try to set 1000
        'checkpoint_every_n_steps': 5000,

    }
}


# In[17]:


flags_dict = {
    "debug": False,
    # --- Data configs ---
    "l5kit_data_folder": "/home/ubuntu/lyft-motion-prediction-autonomous-vehicles/", # change this
    # --- Model configs ---
    "pred_mode": "multi",
    # --- Training configs ---
    "device": "cuda:0", # change this to 'cuba:0' if put on server
    "out_dir": "/home/ubuntu/lyft-output/resnet34-267-0.5-32-2",
    "epoch": 2, 
    "snapshot_freq": 50,
}


# In[18]:


flags = DotDict(flags_dict)
out_dir = Path(flags.out_dir)
os.makedirs(str(out_dir), exist_ok=True)
print(f"flags: {flags_dict}")
save_yaml(out_dir / 'flags.yaml', flags_dict)
save_yaml(out_dir / 'cfg.yaml', cfg)
debug = flags.debu


# In[19]:


#Loading data


# In[20]:


# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = flags.l5kit_data_folder
dm = LocalDataManager(None)

print("Load dataset...")
train_cfg = cfg["train_data_loader"]

# Rasterizer
rasterizer = build_rasterizer(cfg, dm)

# Train dataset/dataloader
def transform(batch):
    return batch["image"], batch["target_positions"], batch["target_availabilities"]


train_path = "scenes/sample.zarr" if debug else train_cfg["key"]
train_zarr = ChunkedDataset(dm.require(train_path)).open()
print("train_zarr", type(train_zarr))
train_agent_dataset = AgentDataset(cfg, train_zarr, rasterizer)
train_dataset = TransformDataset(train_agent_dataset, transform)
if debug:
    # Only use 1000 dataset for fast check...
    train_dataset = Subset(train_dataset, np.arange(1000))
train_loader = DataLoader(train_dataset,
                          shuffle=train_cfg["shuffle"],
                          batch_size=train_cfg["batch_size"],
                          num_workers=train_cfg["num_workers"])
print(train_agent_dataset)
print("# AgentDataset train:", len(train_agent_dataset))
print("# ActualDataset train:", len(train_dataset))


# In[21]:


from typing import Dict

from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50
from tqdm import tqdm

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from prettytable import PrettyTable
from pathlib import Path

import os


# In[22]:


# ===== GENERATE AND LOAD CHOPPED DATASET
valid_cfg = cfg["valid_data_loader"]
valid_path = "scenes/sample.zarr" if debug else valid_cfg["key"]
num_frames_to_chop = 100
MIN_FUTURE_STEPS = 10
valid_base_path = create_chopped_dataset(dm.require(valid_cfg["key"]), cfg["raster_params"]["filter_agents_threshold"], 
                              num_frames_to_chop, cfg["model_params"]["future_num_frames"], MIN_FUTURE_STEPS)


# In[23]:


valid_zarr_path = str(Path(valid_base_path) / Path(dm.require(valid_cfg["key"])).name)
valid_mask_path = str(Path(valid_base_path) / "mask.npz")
valid_gt_path = str(Path(valid_base_path) / "gt.csv")

valid_zarr = ChunkedDataset(valid_zarr_path).open()
valid_mask = np.load(valid_mask_path)["arr_0"]
# ===== INIT DATASET AND LOAD MASK
valid_agent_dataset = AgentDataset(cfg, valid_zarr, rasterizer, agents_mask=valid_mask)
valid_dataset = TransformDataset(valid_agent_dataset, transform)
valid_loader = DataLoader(valid_dataset,
                          shuffle=valid_cfg["shuffle"],
                          batch_size=valid_cfg["batch_size"],
                          num_workers=valid_cfg["num_workers"])

print(valid_agent_dataset)
print("# AgentDataset valid:", len(valid_agent_dataset))


# In[30]:


print("# ActualDataset valid:", len(valid_dataset))


# In[ ]:


#Prepare model & optimizer


# In[26]:


device = torch.device(flags.device)

if flags.pred_mode == "multi":
    predictor = LyftMultiModel(cfg)
    model = LyftMultiRegressor(predictor)
else:
    raise ValueError(f"[ERROR] Unexpected value flags.pred_mode={flags.pred_mode}")

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# In[ ]:


#Write training code
#pytorch-ignite & pytorch-pfn-extras are used here.

#pytorch/ignite: It provides abstraction for writing training loop.
#pfnet/pytorch-pfn-extras: It provides several "extensions" useful for training.
#Useful for logging, printing, evaluating, saving the model, scheduling the learning rate during training.


# In[ ]:


# Train setup
trainer = create_trainer(model, optimizer, device)


def eval_func(*batch):
    loss, metrics = model(*[elem.to(device) for elem in batch])


valid_evaluator = E.Evaluator(
    valid_loader,
    model,
    progress_bar=True,
    eval_func=eval_func,
)

log_trigger = (10 if debug else 1000, "iteration")
log_report = E.LogReport(trigger=log_trigger)


extensions = [
    log_report,  # Save `log` to file
    valid_evaluator,  # Run evaluation for valid dataset in each epoch.
    # E.FailOnNonNumber()  # Stop training when nan is detected.
]

is_notebook = False  # Make it False when you run code in local machine using console.
if is_notebook:
    extensions.extend([
        ProgressBarNotebook(update_interval=10 if debug else 100),  # Show progress bar during training
        PrintReportNotebook(),  # Show "log" on jupyter notebook  
    ])
else:
    extensions.extend([
        E.ProgressBar(update_interval=10 if debug else 100),  # Show progress bar during training
        E.PrintReport(),  # Print "log" to terminal
    ])


epoch = flags.epoch

models = {"main": model}
optimizers = {"main": optimizer}
manager = IgniteExtensionsManager(
    trainer,
    models,
    optimizers,
    epoch,
    extensions=extensions,
    out_dir=str(out_dir),
)
# Save predictor.pt every epoch
manager.extend(E.snapshot_object(predictor, "predictor.pt"),
               trigger=(flags.snapshot_freq, "iteration"))
# Check & Save best validation predictor.pt every epoch
# manager.extend(E.snapshot_object(predictor, "best_predictor.pt"),
#                trigger=MinValueTrigger("validation/main/nll", trigger=(flags.snapshot_freq, "iteration")))
# --- lr scheduler ---
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-10)
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=0.99999)
manager.extend(lambda manager: scheduler.step(), trigger=(1, "iteration"))
# Show "lr" column in log
manager.extend(E.observe_lr(optimizer=optimizer), trigger=log_trigger)

trainer.run(train_loader, max_epochs=epoch)


# In[ ]:


df = log_report.to_dataframe()
df.to_csv("log.csv", index=False)
df[["epoch", "iteration", "main/loss", "main/nll", "validation/main/loss", "validation/main/nll", "lr", "elapsed_time"]]


# In[ ]:


#prediction


# In[20]:


device = torch.device(flags.device)

if flags.pred_mode == "multi":
    predictor = LyftMultiModel(cfg)
else:
    raise ValueError(f"[ERROR] Unexpected value flags.pred_mode={flags.pred_mode}")

pt_path = "/home/ubuntu/lyft-output/resnet34-267-0.5-32-2/predictor.pt" 
# change the file for pt
print(f"Loading from {pt_path}")
predictor.load_state_dict(torch.load(pt_path,map_location=torch.device('cpu')))
predictor.to(device)


# In[22]:


def run_prediction(predictor, data_loader):
    predictor.eval()

    pred_coords_list = []
    confidences_list = []
    timestamps_list = []
    track_id_list = []

    with torch.no_grad():
        dataiter = tqdm(data_loader)
        for data in dataiter:
            image = data["image"].to(device)
            # target_availabilities = data["target_availabilities"].to(device)
            # targets = data["target_positions"].to(device)
            preds, confidences = predictor(image)
            # convert agent coordinates into world offsets
            preds = preds.cpu().numpy().copy()
            world_from_agents = data["world_from_agent"].numpy()
            centroids = data["centroid"].numpy()
            coords_offset = []
        
        # convert into world coordinates and compute offsets
            for idx in range(len(preds)):
                for mode in range(3):
                    preds[idx, mode, :, :] = transform_points(preds[idx, mode, :, :], world_from_agents[idx]) - centroids[idx][:2]
            
            confidences_list.append(confidences.cpu().numpy().copy())
            timestamps_list.append(data["timestamp"].numpy().copy())
            track_id_list.append(data["track_id"].numpy().copy())
            pred_coords_list.append(preds.copy())
    timestamps = np.concatenate(timestamps_list)
    track_ids = np.concatenate(track_id_list)
    coords = np.concatenate(pred_coords_list)
    confs = np.concatenate(confidences_list)
    return timestamps, track_ids, coords, confs


# In[27]:


# set env variable for data
l5kit_data_folder = "/home/ubuntu/lyft-motion-prediction-autonomous-vehicles"
os.environ["L5KIT_DATA_FOLDER"] = l5kit_data_folder
dm = LocalDataManager(None)

print("Load dataset...")
default_test_cfg = {
    'key': 'scenes/test.zarr',
    'batch_size': 32,
    'shuffle': False,
    'num_workers': 4
}
test_cfg = cfg.get("test_data_loader", default_test_cfg)

# Rasterizer
rasterizer = build_rasterizer(cfg, dm)

test_path = test_cfg["key"]
print(f"Loading from {test_path}")
test_zarr = ChunkedDataset(dm.require(test_path)).open()
print("test_zarr", type(test_zarr))
test_mask = np.load(f"{l5kit_data_folder}/scenes/mask.npz")["arr_0"]
test_agent_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
test_dataset = test_agent_dataset
if debug:
    # Only use 100 dataset for fast check...
    test_dataset = Subset(test_dataset, np.arange(100))
test_loader = DataLoader(
    test_dataset,
    shuffle=test_cfg["shuffle"],
    batch_size=test_cfg["batch_size"],
    num_workers=test_cfg["num_workers"],
    pin_memory=True,
)

print(test_agent_dataset)
print("# AgentDataset test:", len(test_agent_dataset))
print("# ActualDataset test:", len(test_dataset))


# In[26]:


timestamps, track_ids, coords, confs = run_prediction(predictor, test_loader)


# In[27]:


csv_path = "resnet34-267-0.5-32-2.csv"
write_pred_csv(
    csv_path,
    timestamps=timestamps,
    track_ids=track_ids,
    coords=coords,
    confs=confs)
print(f"Saved to {csv_path}")


# In[ ]:




