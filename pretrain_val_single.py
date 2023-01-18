import pandas as pd
import torch
import numpy as np
import json
import importlib
import os

from torch.utils.data import DataLoader

from configs.project_config import ProjectConfig
from configs.data_shape_config import DataShapeConfig
from configs.run_config.pretrain_config import PretrainConfig, PUBTestConfig, ValConfig
from configs.run_config.fine_tune_config import FineTuneConfig
from pretrain import n_epochs
from utils.eval_model import eval_model_obs_preds
from utils.metrics import calc_mse, calc_nse
from utils.tools import SeedMethods
from utils.test_full import test_full
from data.dataset import DatasetFactory

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
# pre_test_config = PretrainConfig.pre_test_config
batch_size = PretrainConfig.batch_size

exps_config = ValConfig.exps_config  # CHANGE：原站点val测试

if __name__ == '__main__':
    print("pid:", os.getpid())
    SeedMethods.seed_torch(seed=seed)
    print(saving_root)
    # Model
    # # Define model type
    models = importlib.import_module("models")
    Model = getattr(models, used_model)
    # best_path = list(saving_root.glob(f"(max_nse)*.pkl"))
    # assert (len(best_path) == 1)
    # best_path = best_path[0]
    # cur_model = Model().to(device)
    # best_model.load_state_dict(torch.load(best_path, map_location=device))

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

    nse_val = pd.DataFrame(columns=['epoch', 'nse_median', 'nse_mean', 'mse_median', 'mse_mean'])
    nse_basin_dict = dict()
    mse_basin_dict = dict()
    nse_mean_best = {'epoch': -1, 'val': -100}
    mse_mean_best = {'epoch': -1, 'val': -100}

    exps_num = len(exps_config)
    for idx, exp_config in enumerate(exps_config):
        print(f"==========Now process: {idx} / {exps_num}===========")
        SeedMethods.seed_torch(seed=seed)
        # root_now = saving_root / "pub_val_single" / exp_config["tag"]
        # root_now.mkdir(parents=True, exist_ok=True)

        # Testing data (needs training mean and training std)
        ds_val = DS.get_instance(past_len, pred_len, "val", specific_cfg=exp_config["ft_val_config"],
                                 x_mean=train_x_mean, y_mean=train_y_mean, x_std=train_x_std, y_std=train_y_std,
                                 y_stds_dict=y_stds_dict)
        val_loader = DataLoader(ds_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        date_index = val_loader.dataset.date_index_dict[exp_config["tag"]]

        nse_basin = []
        mse_basin = []
        for epoch in range(n_epochs):
            # print(f"-----{exp_config['tag']}:{epoch}-----")
            cur_path = list(saving_root.glob(f"(epoch)_{epoch}_{epoch}.pkl"))
            assert (len(cur_path) == 1)
            cur_path = cur_path[0]
            cur_model = Model().to(device)
            cur_model.load_state_dict(torch.load(cur_path, map_location=device))

            obs, pred = eval_model_obs_preds(cur_model, val_loader, decode_mode, device)
            # Calculate nse after rescale (But if you take the same mean and std, it's equivalent before and after)
            obs = obs.numpy()
            pred = pred.numpy()
            _, mses_test = calc_mse(obs, pred)
            test_mse_mean = mses_test.mean()

            obs_rescaled = val_loader.dataset.local_rescale(obs, variable='output')
            pred_rescaled = val_loader.dataset.local_rescale(pred, variable='output')

            # pred_rescaled[pred_rescaled < 0] = 0  # ADD

            _, nses_test = calc_nse(obs_rescaled, pred_rescaled)
            test_nse_mean = nses_test.mean()
            # test_mse, test_nse = test_full(cur_model, decode_mode, test_loader, device, root_now, False,
            #                                date_index=date_index)
            nse_basin.append(test_nse_mean)
            mse_basin.append(test_mse_mean)

        nse_basin_dict[exp_config["tag"]] = nse_basin
        mse_basin_dict[exp_config["tag"]] = mse_basin
        print(f'nse_basin_dict[{exp_config["tag"]}]:', nse_basin)
        print(f'mse_basin_dict[{exp_config["tag"]}]:', mse_basin)

    nse_basins_value = np.vstack(list(nse_basin_dict.values()))
    nse_median = np.median(nse_basins_value, axis=0)
    nse_mean = np.mean(nse_basins_value, axis=0)
    nse_mean_best['val'] = np.max(nse_mean)
    nse_mean_best['epoch'] = list(nse_mean).index(nse_mean_best['val'])
    print("nse mean best:", nse_mean_best)
    print("nse median best:", np.max(nse_median),list(nse_median).index(np.max(nse_median)))
    mse_basins_value = np.vstack(list(mse_basin_dict.values()))
    mse_median = np.median(mse_basins_value, axis=0)
    mse_mean = np.mean(mse_basins_value, axis=0)
    mse_mean_best['val'] = np.min(mse_mean)
    mse_mean_best['epoch'] = list(mse_mean).index(mse_mean_best['val'])
    print("mse mean best:", mse_mean_best)

    for epoch in range(n_epochs):
        nse_val = nse_val.append(
            pd.DataFrame({'epoch': [epoch], 'nse_median': [nse_median[epoch]], 'nse_mean': [nse_mean[epoch]],
                          'mse_median': [mse_median[epoch]], 'mse_mean': [mse_mean[epoch]]}))
    print(nse_val)
    nse_val.to_csv(saving_root / "nse_val.txt", encoding="utf-8")
