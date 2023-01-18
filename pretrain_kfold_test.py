import pandas as pd
import torch
import os
import numpy as np
import json
import importlib
from shutil import copytree
from torch.utils.data import DataLoader

from configs.project_config import ProjectConfig
from configs.data_shape_config import DataShapeConfig
from configs.run_config.pretrain_config import KFoldConfig, KFoldTestConfig
from pretrain import batch_size
from utils.test_full import test_full
from utils.tools import SeedMethods
from utils.lr_strategies import SchedulerFactory
from utils.train_full import train_full
from data.dataset import DatasetFactory
from sklearn.model_selection import KFold

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

used_model = KFoldTestConfig.used_model
decode_mode = KFoldTestConfig.decode_mode

exps_config = KFoldTestConfig.exps_config
split = KFoldTestConfig.split

seed = KFoldTestConfig.seed
# saving_message = KFoldConfig.saving_message
# saving_root = KFoldConfig.saving_root
pre_saving_root = KFoldTestConfig.pre_saving_root

if __name__ == '__main__':
    print("pid:", os.getpid())
    SeedMethods.seed_torch(seed=seed)
    print(f"-------------Cross global test {split}-------------")
    # saving_root = pre_saving_root / str(split)
    saving_root = pre_saving_root
    print(saving_root)

    # Define model type
    models = importlib.import_module("models")
    Model = getattr(models, used_model)
    # Model
    best_name = f"(max_nse)*.pkl"
    # best_name = f"(epoch)_198*.pkl"
    best_path = list(saving_root.glob(best_name))
    assert (len(best_path) == 1)
    best_path = best_path[0]
    best_model = Model().to(device)
    best_model.load_state_dict(torch.load(best_path, map_location=device))

    # Dataset
    DS = DatasetFactory.get_dataset_type(use_future_fea, use_static)

    train_means = np.loadtxt(saving_root / "train_means.csv", dtype="float32")
    train_stds = np.loadtxt(saving_root / "train_stds.csv", dtype="float32")
    train_x_mean = train_means[:-1]
    train_y_mean = train_means[-1]
    train_x_std = train_stds[:-1]
    train_y_std = train_stds[-1]
    with open(saving_root / "y_stds_dict.json", "rt") as f:
        y_stds_dict = json.load(f)

    metric_results = pd.DataFrame(columns=['basin', 'nse', 'rmse', 'kge'])
    exps_num = len(exps_config)
    for idx, exp_config in enumerate(exps_config):
        print(f"==========Now process: {idx} / {exps_num}===========")
        SeedMethods.seed_torch(seed=seed)
        root_now = saving_root / f"test_single" / f'{exp_config["tag"]}_{best_name}'
        root_now.mkdir(parents=True, exist_ok=True)
        # Testing data (needs training mean and training std)
        ds_test = DS.get_instance(past_len, pred_len, "test", specific_cfg=exp_config["pre_test_config"],
                                  x_mean=train_x_mean, y_mean=train_y_mean, x_std=train_x_std, y_std=train_y_std,
                                  y_stds_dict=y_stds_dict)
        # ds_test = DS.get_instance(past_len, pred_len, "test", specific_cfg=exp_config["ft_test_config"],
        #                           x_mean=train_x_mean, y_mean=train_y_mean, x_std=train_x_std, y_std=train_y_std,
        #                           y_stds_dict=None)  # CHANGE:PUB似乎不需要y_stds_dict
        test_loader = DataLoader(ds_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        date_index = test_loader.dataset.date_index_dict[exp_config["tag"]]

        test_rmse, test_nse, test_kge = test_full(best_model, decode_mode, test_loader, device, root_now, False,
                                                  date_index=date_index)
        # break   #NOTE:临时

        # ADD
        basin = exp_config["tag"]
        metric_results = metric_results.append(
            pd.DataFrame({'basin': [basin], 'nse': [test_nse], 'rmse': [test_rmse], 'kge': [test_kge]}))

    result_dir = saving_root / f"test_result_{best_name}"
    if not os.path.isdir(result_dir):
        result_dir.mkdir(parents=True)
    metric_results.to_csv(result_dir / f"test_all.txt",
                          encoding="utf-8",
                          sep=',', index=False)

    results_value = metric_results.values
    results_nse = results_value[:, 1]
    results_rmse = results_value[:, 2]
    results_kge = results_value[:, 3]
    with open(result_dir / f"test_statistics_{best_name}.txt", 'w+') as f:
        f.write(f"nse_median:{np.median(results_nse)},nse_mean:{np.mean(results_nse)}\n")
        f.write(f"rmse_median:{np.median(results_rmse)},rmse_mean:{np.mean(results_rmse)}\n")
        f.write(f"kge_median:{np.median(results_kge)},kge_mean:{np.mean(results_kge)}\n")
    print(f"nse_median:{np.median(results_nse)},nse_mean:{np.mean(results_nse)}\n",
          f"rmse_median:{np.median(results_rmse)},rmse_mean:{np.mean(results_rmse)}\n",
          f"kge_median:{np.median(results_kge)},kge_mean:{np.mean(results_kge)}\n")
