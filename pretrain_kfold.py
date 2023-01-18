import warnings

import torch
import os
import numpy as np
import json
import importlib
from shutil import copytree
from torch.utils.data import DataLoader

from configs.project_config import ProjectConfig
from configs.data_shape_config import DataShapeConfig
from configs.run_config.pretrain_config import KFoldConfig
from utils.tools import SeedMethods
from utils.lr_strategies import SchedulerFactory
from utils.train_full import train_full
from data.dataset import DatasetFactory

project_root = ProjectConfig.project_root
device = ProjectConfig.device
num_workers = ProjectConfig.num_workers

past_len = DataShapeConfig.past_len
pred_len = DataShapeConfig.pred_len
src_len = DataShapeConfig.src_len
tgt_len = DataShapeConfig.tgt_len
src_size = DataShapeConfig.src_size
tgt_size = DataShapeConfig.tgt_size
use_future_fea = DataShapeConfig.use_future_fea
use_static = DataShapeConfig.use_static

used_model = KFoldConfig.used_model
decode_mode = KFoldConfig.decode_mode
loss_func = KFoldConfig.loss_func
n_epochs = KFoldConfig.n_epochs
batch_size = KFoldConfig.batch_size
learning_rate = KFoldConfig.learning_rate
scheduler_paras = KFoldConfig.scheduler_paras

pre_train_config = KFoldConfig.pre_train_config
pre_val_config = KFoldConfig.pre_val_config
pre_test_config = KFoldConfig.pre_test_config
split = KFoldConfig.split

seed = KFoldConfig.seed
saving_message = KFoldConfig.saving_message
saving_root = KFoldConfig.saving_root

if __name__ == '__main__':
    print("pid:", os.getpid())
    SeedMethods.seed_torch(seed=seed)

    # saving_root.mkdir(exist_ok=True, parents=True)
    # print(saving_root)
    print(f"-------------Cross vali {split}-------------")
    saving_root = saving_root / str(split)
    saving_root.mkdir(exist_ok=True, parents=True)
    print(saving_root)

    # Saving config files
    configs_path = project_root / "configs"
    configs_saving = saving_root / "configs"
    if configs_saving.exists():
        # raise RuntimeError("config files already exists!")
        warnings.warn("config files already exists!")
    else:
        copytree(configs_path, configs_saving)

    # Define model type
    models = importlib.import_module("models")
    Model = getattr(models, used_model)
    # # for continue
    # continue_name = f"(epoch)*.pkl"
    # continue_path = list(saving_root.glob(continue_name))
    # assert (len(continue_path) == 1)
    # continue_path = continue_path[0]
    # continue_model = Model().to(device)
    # continue_model.load_state_dict(torch.load(continue_path, map_location=device))
    # # for continue

    # Dataset
    DS = DatasetFactory.get_dataset_type(use_future_fea, use_static)

    # Training data
    ds_train = DS.get_instance(past_len, pred_len, "train", specific_cfg=pre_train_config)
    train_loader = DataLoader(ds_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    # We use the feature means/stds of the training data for normalization in val and test stage
    train_x_mean, train_y_mean = ds_train.get_means()
    train_x_std, train_y_std = ds_train.get_stds()
    y_stds_dict = ds_train.y_stds_dict
    # Saving training mean and training std
    train_means = np.concatenate((train_x_mean, train_y_mean), axis=0)
    train_stds = np.concatenate((train_x_std, train_y_std), axis=0)
    np.savetxt(saving_root / "train_means.csv", train_means)
    np.savetxt(saving_root / "train_stds.csv", train_stds)
    with open(saving_root / "y_stds_dict.json", "wt") as f:
        json.dump(y_stds_dict, f)

    # # Validation data (needs training mean and training std)
    ds_val = DS.get_instance(past_len, pred_len, "val", specific_cfg=pre_val_config,
                             x_mean=train_x_mean, y_mean=train_y_mean, x_std=train_x_std, y_std=train_y_std,
                             y_stds_dict=y_stds_dict)
    val_loader = DataLoader(ds_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # Testing data (needs training mean and training std)
    ds_test = DS.get_instance(past_len, pred_len, "test", specific_cfg=pre_test_config,
                              x_mean=train_x_mean, y_mean=train_y_mean, x_std=train_x_std, y_std=train_y_std,
                              y_stds_dict=y_stds_dict)
    test_loader = DataLoader(ds_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # Model, Optimizer, Scheduler, Loss
    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = SchedulerFactory.get_scheduler(optimizer, **scheduler_paras)
    loss_func = loss_func.to(device)

    # Training and Validation
    train_full(model, decode_mode, train_loader, val_loader, optimizer, scheduler, loss_func, n_epochs, device,
               saving_root)
