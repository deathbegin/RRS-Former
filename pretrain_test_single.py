import pandas as pd
import torch
import numpy as np
import json
import importlib
import os

from torch.utils.data import DataLoader

from configs.project_config import ProjectConfig
from configs.data_shape_config import DataShapeConfig
from configs.run_config.pretrain_config import PretrainConfig, PUBTestConfig
from configs.run_config.fine_tune_config import FineTuneConfig
from utils.tools import SeedMethods
from utils.test_full import test_full
from data.dataset import DatasetFactory
from configs.dataset_config import DatasetConfig

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

seed = PretrainConfig.seed
saving_message = PretrainConfig.saving_message
saving_root = PretrainConfig.saving_root
used_model = PretrainConfig.used_model
decode_mode = PretrainConfig.decode_mode
pre_test_config = PretrainConfig.pre_test_config
batch_size = PretrainConfig.batch_size
basins_test_mark = DatasetConfig.basins_test_mark

# exps_config = FineTuneConfig.exps_config # CHANGE：普通测试
exps_config = PUBTestConfig.exps_config  # ADD：PUB测试

if __name__ == '__main__':
    print("pid:", os.getpid())
    SeedMethods.seed_torch(seed=seed)
    print(saving_root)
    # Model
    # # Define model type
    models = importlib.import_module("models")
    Model = getattr(models, used_model)
    best_epoch = "(max_nse)*.pkl"  # NOTE:default="(max_nse)*.pkl"
    # best_epoch = "(epoch)_7_*.pkl"  # NOTE:default="(max_nse)*.pkl"
    # best_path = list(saving_root.glob(f"(max_nse)*.pkl"))
    best_path = list(saving_root.glob(f"{best_epoch}"))
    assert (len(best_path) == 1)
    best_path = best_path[0]
    best_model = Model().to(device)
    best_model.load_state_dict(torch.load(best_path, map_location=device))

    # Dataset
    DS = DatasetFactory.get_dataset_type(use_future_fea, use_static)
    # # Needs training mean and training std
    train_means = np.loadtxt(saving_root / "train_means.csv", dtype="float32")
    train_stds = np.loadtxt(saving_root / "train_stds.csv", dtype="float32")
    train_x_mean = train_means[:-1]
    train_y_mean = train_means[-1]
    train_x_std = train_stds[:-1]
    train_y_std = train_stds[-1]
    with open(saving_root / "y_stds_dict.json", "rt") as f:
        y_stds_dict = json.load(f)

    # mse_all = []
    # nse_all = []
    # kge_all = []
    metric_results = pd.DataFrame(columns=['basin', 'nse', 'rmse', 'kge'])
    exps_num = len(exps_config)
    for idx, exp_config in enumerate(exps_config):
        print(f"==========Now process: {idx} / {exps_num}===========")
        SeedMethods.seed_torch(seed=seed)
        root_now = saving_root / f"pub_test_single_{basins_test_mark}_w0" / exp_config["tag"]
        root_now.mkdir(parents=True, exist_ok=True)
        # Testing data (needs training mean and training std)
        ds_test = DS.get_instance(past_len, pred_len, "test", specific_cfg=exp_config["ft_test_config"],
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

    result_dir = saving_root / f"test_{basins_test_mark}_result_w0"
    if not os.path.isdir(result_dir):
        result_dir.mkdir(parents=True)
    metric_results.to_csv(result_dir / f"test_all_{best_epoch}.txt",
                          encoding="utf-8",
                          sep=',', index=False)

    results_value = metric_results.values
    results_nse = results_value[:, 1]
    results_rmse = results_value[:, 2]
    results_kge = results_value[:, 3]
    with open(result_dir / f"test_statistics_{best_epoch}.txt", 'w+') as f:
        f.write(f"nse_median:{np.median(results_nse)},nse_mean:{np.mean(results_nse)}\n")
        f.write(f"rmse_median:{np.median(results_rmse)},rmse_mean:{np.mean(results_rmse)}\n")
        f.write(f"kge_median:{np.median(results_kge)},kge_mean:{np.mean(results_kge)}\n")
    print(f"nse_median:{np.median(results_nse)},nse_mean:{np.mean(results_nse)}\n",
          f"rmse_median:{np.median(results_rmse)},rmse_mean:{np.mean(results_rmse)}\n",
          f"kge_median:{np.median(results_kge)},kge_mean:{np.mean(results_kge)}\n")


    # mse_all.append(test_mse)
    # nse_all.append(test_nse)

    # nse_mean = np.mean(nse_all)
    # nse_median = np.median(nse_all)
    # print(nse_median, nse_mean)
    # nse_all.insert(0, nse_mean)  # NOTE:第二个是均值
    # nse_all.insert(0, nse_median)  # NOTE:打头是中值
    # # print(nse_all)
    # nse_all = pd.DataFrame(data=nse_all)
    # nse_all.to_csv(saving_root / f"nse_test_{best_epoch}.txt", encoding="utf-8", sep='\t', index=False, header=False)
    #
    # mse_mean = np.mean(mse_all)
    # mse_median = np.median(mse_all)
    # print(mse_median, mse_mean)
    # mse_all.insert(0, mse_mean)  # NOTE:第二个是均值
    # mse_all.insert(0, mse_median)  # NOTE:打头是中值
    # # print(mse_all)
    # mse_all = pd.DataFrame(data=mse_all)
    # mse_all.to_csv(saving_root / f"mse_test_{best_epoch}.txt", encoding="utf-8", sep='\t', index=False, header=False)
