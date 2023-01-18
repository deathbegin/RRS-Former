import numpy as np
import torch.nn as nn
import importlib

from sklearn.model_selection import KFold

from ..project_config import ProjectConfig
from ..dataset_config import DatasetConfig
from ..data_shape_config import DataShapeConfig
from ..model_config.Transformer_config import TransformerConfig
from utils import nseloss


class PretrainLearningConfig:
    loss_type = "NSELoss"  # TODO: loss function type, chose in ["NSELoss" ,"MSE"]
    loss_functions = {"MSE": nn.MSELoss(), "NSELoss": nseloss.NSELoss()}
    loss_func = loss_functions[loss_type]

    scale_factor = 1  # TODO: usually, the bath_size bigger is, the learning_rate larger will have to be.
    n_epochs = 200  # TODO:origin:200
    batch_size = 512 // scale_factor  # TODO
    learning_rate = 0.001 / scale_factor / 1  # TEST:lr=0.0002
    # "type" chose in [none, warm_up, cos_anneal, exp_decay]  # TODO
    scheduler_paras = {"scheduler_type": "warm_up", "last_epoch": -1, "warm_up_epochs": n_epochs * 0.25, "decay_rate":
        0.99}
    # scheduler_paras = {"scheduler_type": "warm_up", "warm_up_epochs": n_epochs * 0.25, "decay_rate": 0.5}
    # scheduler_paras = {"scheduler_type": "none"}
    # scheduler_paras = {"scheduler_type": "exp_decay", "decay_epoch": n_epochs * 0.5, "decay_rate": 0.99}
    # scheduler_paras = {"scheduler_type": "cos_anneal", "cos_anneal_t_max": 32}

    learning_config_info = f"{loss_type}_n{n_epochs}_bs{batch_size}_lr{learning_rate}_{scheduler_paras['scheduler_type']}"


class PretrainConfig(PretrainLearningConfig):
    seed = 1234  # Random seed
    used_model = "Transformer"  # TODO

    used_model_config = importlib.import_module(f"configs.model_config.{used_model}_config")
    used_ModelConfig = getattr(used_model_config, f"{used_model}Config")
    decode_mode = used_ModelConfig.decode_mode
    model_info = used_ModelConfig.model_info

    # pre_train_id = f"{DatasetConfig.forcing_type}{DatasetConfig.basin_mark}@{DataShapeConfig.data_shape_info}" \
    #                f"@{DatasetConfig.train_start.date()}~{DatasetConfig.train_end.date()}"
    # pre_val_id = f"{DatasetConfig.forcing_type}{DatasetConfig.basin_mark}@{DataShapeConfig.data_shape_info}" \
    #              f"@{DatasetConfig.val_start.date()}~{DatasetConfig.val_end.date()}"
    # pre_test_id = f"{DatasetConfig.forcing_type}{DatasetConfig.basin_mark}@{DataShapeConfig.data_shape_info}" \
    #               f"@{DatasetConfig.test_start.date()}~{DatasetConfig.test_end.date()}"
    pre_train_id = f"{DatasetConfig.forcing_type}_{DatasetConfig.huc}_{DatasetConfig.basin_mark}@{DataShapeConfig.data_shape_info}" \
                   f"@{DatasetConfig.train_start.date()}~{DatasetConfig.train_end.date()}"
    pre_val_id = f"{DatasetConfig.forcing_type}_{DatasetConfig.huc}_{DatasetConfig.basin_mark}@{DataShapeConfig.data_shape_info}" \
                 f"@{DatasetConfig.val_start.date()}~{DatasetConfig.val_end.date()}"
    pre_test_id = f"{DatasetConfig.forcing_type}_{DatasetConfig.huc}_{DatasetConfig.basin_mark}@{DataShapeConfig.data_shape_info}" \
                  f"@{DatasetConfig.test_start.date()}~{DatasetConfig.test_end.date()}"
    final_train_data_path = ProjectConfig.final_data_root / f"{pre_train_id}_serialized_train.pkl"
    final_val_data_path = ProjectConfig.final_data_root / f"{pre_val_id}_serialized_val.pkl"
    final_test_data_path = ProjectConfig.final_data_root / f"{pre_test_id}_serialized_test.pkl"
    pre_train_config = {
        "camels_root": DatasetConfig.camels_root,
        "basins_list": DatasetConfig.global_basins_list,
        "forcing_type": DatasetConfig.forcing_type,
        "start_date": DatasetConfig.train_start,
        "end_date": DatasetConfig.train_end,
        "use_runoff": DataShapeConfig.use_runoff,
        "final_data_path": final_train_data_path
    }

    pre_val_config = {
        "camels_root": DatasetConfig.camels_root,
        "basins_list": DatasetConfig.global_basins_list,
        "forcing_type": DatasetConfig.forcing_type,
        "start_date": DatasetConfig.val_start,
        "end_date": DatasetConfig.val_end,
        "use_runoff": DataShapeConfig.use_runoff,
        "final_data_path": final_val_data_path
    }

    pre_test_config = {
        "camels_root": DatasetConfig.camels_root,
        "basins_list": DatasetConfig.global_basins_list,
        "forcing_type": DatasetConfig.forcing_type,
        "start_date": DatasetConfig.test_start,
        "end_date": DatasetConfig.test_end,
        "use_runoff": DataShapeConfig.use_runoff,
        "final_data_path": final_test_data_path
    }

    # TODO:for test search
    # saving_message = f"{model_info}@{DatasetConfig.dataset_info}@{DataShapeConfig.data_shape_info}" \
    #                  f"@{PretrainLearningConfig.learning_config_info}@dp{TransformerConfig.dropout_rate}@seed{seed}"
    # TODO:for origin，test_decoder,test_upper
    saving_message = f"{model_info}@{DatasetConfig.dataset_info}@{DataShapeConfig.data_shape_info}" \
                     f"@{PretrainLearningConfig.learning_config_info}@seed{seed}"

    saving_root = ProjectConfig.run_root / saving_message


class KFoldConfig(PretrainLearningConfig):
    seed = 1234  # Random seed
    used_model = "Transformer"  # TODO
    split = 4  # NOTE:in [0,...,11],[0,1,2,3,4]

    used_model_config = importlib.import_module(f"configs.model_config.{used_model}_config")
    used_ModelConfig = getattr(used_model_config, f"{used_model}Config")
    decode_mode = used_ModelConfig.decode_mode
    model_info = used_ModelConfig.model_info

    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    global_basins_nplist = np.array(DatasetConfig.global_basins_list)
    train_idx, test_idx = list(kfold.split(global_basins_nplist))[split]
    train_basins_list, test_basins_list = global_basins_nplist[train_idx], global_basins_nplist[test_idx]

    # pre_train_id = f"{DatasetConfig.forcing_type}{DatasetConfig.basin_mark}@{DataShapeConfig.data_shape_info}" \
    #                f"@{DatasetConfig.train_start.date()}~{DatasetConfig.train_end.date()}"
    # pre_val_id = f"{DatasetConfig.forcing_type}{DatasetConfig.basin_mark}@{DataShapeConfig.data_shape_info}" \
    #              f"@{DatasetConfig.val_start.date()}~{DatasetConfig.val_end.date()}"
    # pre_test_id = f"{DatasetConfig.forcing_type}{DatasetConfig.basin_mark}@{DataShapeConfig.data_shape_info}" \
    #               f"@{DatasetConfig.test_start.date()}~{DatasetConfig.test_end.date()}"
    # NOTE:CHANGE
    pre_train_id = f"{DatasetConfig.forcing_type}_{DatasetConfig.huc}_{DatasetConfig.basin_mark}@{DataShapeConfig.data_shape_info}" \
                   f"@{DatasetConfig.train_start.date()}~{DatasetConfig.train_end.date()}~{split}"
    pre_val_id = f"{DatasetConfig.forcing_type}_{DatasetConfig.huc}_{DatasetConfig.basin_mark}@{DataShapeConfig.data_shape_info}" \
                 f"@{DatasetConfig.val_start.date()}~{DatasetConfig.val_end.date()}~{split}"
    pre_test_id = f"{DatasetConfig.forcing_type}_{DatasetConfig.huc}_{DatasetConfig.basin_mark}@{DataShapeConfig.data_shape_info}" \
                  f"@{DatasetConfig.test_start.date()}~{DatasetConfig.test_end.date()}~{split}"
    final_train_data_path = ProjectConfig.final_data_root / f"{pre_train_id}_serialized_train.pkl"
    final_val_data_path = ProjectConfig.final_data_root / f"{pre_val_id}_serialized_val.pkl"
    final_test_data_path = ProjectConfig.final_data_root / f"{pre_test_id}_serialized_test.pkl"

    pre_train_config = {  # NOTE:CHANGE
        "camels_root": DatasetConfig.camels_root,
        "basins_list": train_basins_list,  # NOTE:CHANGE
        "forcing_type": DatasetConfig.forcing_type,
        "start_date": DatasetConfig.train_start,
        "end_date": DatasetConfig.train_end,
        "use_runoff": DataShapeConfig.use_runoff,
        "final_data_path": final_train_data_path
    }

    pre_val_config = {
        "camels_root": DatasetConfig.camels_root,
        "basins_list": train_basins_list,  # NOTE:CHANGE
        "forcing_type": DatasetConfig.forcing_type,
        "start_date": DatasetConfig.val_start,
        "end_date": DatasetConfig.val_end,
        "use_runoff": DataShapeConfig.use_runoff,
        "final_data_path": final_val_data_path
    }

    pre_test_config = {  # NOTE:CHANGE
        "camels_root": DatasetConfig.camels_root,
        "basins_list": test_basins_list,  # NOTE:CHANGE
        "forcing_type": DatasetConfig.forcing_type,
        "start_date": DatasetConfig.test_start,
        "end_date": DatasetConfig.test_end,
        "use_runoff": DataShapeConfig.use_runoff,
        "final_data_path": final_test_data_path
    }

    # TODO:for test search
    # saving_message = f"{model_info}@{DatasetConfig.dataset_info}@{DataShapeConfig.data_shape_info}" \
    #                  f"@{PretrainLearningConfig.learning_config_info}@dp{TransformerConfig.dropout_rate}@seed{seed}"
    # TODO:for origin，test_decoder,test_upper
    saving_message = f"{model_info}@{DatasetConfig.dataset_info}@{DataShapeConfig.data_shape_info}" \
                     f"@{PretrainLearningConfig.learning_config_info}@seed{seed}"

    saving_root = ProjectConfig.run_root / saving_message


class KFoldTestConfig(PretrainLearningConfig):
    seed = 1234  # Random seed
    pre_saving_message = "/home/zhuwu/PUB/RR-Former/runs/4huc_test_decoder_new/Transformer_NAR_[64-4-4-256-0.1]@daymet_4huc_241basins_list_1980~1995#1995~2000#2000~2014@15|14+1[32+1]@NSELoss_n200_bs512_lr0.001_warm_up@seed1234"
    used_model = "Transformer"  # TODO
    split = 2  # NOTE:in [0,...,4]

    used_model_config = importlib.import_module(f"configs.model_config.{used_model}_config")
    used_ModelConfig = getattr(used_model_config, f"{used_model}Config")
    decode_mode = used_ModelConfig.decode_mode
    model_info = used_ModelConfig.model_info

    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    global_basins_nplist = np.array(DatasetConfig.global_basins_list)
    train_idx, test_idx = list(kfold.split(global_basins_nplist))[split]
    test_basins_list = global_basins_nplist[test_idx]

    exps_config = list()
    for basin in test_basins_list:
        exp_config = dict()
        exp_config["tag"] = basin

        exp_config["pre_train_config"] = {  # NOTE:CHANGE
            "camels_root": DatasetConfig.camels_root,
            "basins_list": [basin],  # NOTE:CHANGE
            "forcing_type": DatasetConfig.forcing_type,
            "start_date": DatasetConfig.train_start,
            "end_date": DatasetConfig.train_end,
            "use_runoff": DataShapeConfig.use_runoff,
            "final_data_path": None
        }

        # exp_config['pre_val_config'] = {
        #     "camels_root": DatasetConfig.camels_root,
        #     "basins_list": DatasetConfig.global_basins_list,
        #     "forcing_type": DatasetConfig.forcing_type,
        #     "start_date": DatasetConfig.val_start,
        #     "end_date": DatasetConfig.val_end,
        #     "use_runoff": DataShapeConfig.use_runoff,
        #     "final_data_path": final_val_data_path
        # }

        exp_config['pre_test_config'] = {  # NOTE:CHANGE
            "camels_root": DatasetConfig.camels_root,
            "basins_list": [basin],  # NOTE:CHANGE
            "forcing_type": DatasetConfig.forcing_type,
            "start_date": DatasetConfig.test_start,
            "end_date": DatasetConfig.test_end,
            "use_runoff": DataShapeConfig.use_runoff,
            "final_data_path": None
        }
        exps_config.append(exp_config)

    if pre_saving_message != "":
        pre_saving_root = ProjectConfig.run_root / pre_saving_message / str(split)
        pre_model_path = list(pre_saving_root.glob(f"(max_nse)*.pkl"))
        assert (len(pre_model_path) == 1)
        pre_model_path = pre_model_path[0]
        test_root = pre_saving_root / "test"


class PUBTestConfig(PretrainLearningConfig):
    seed = 1234  # Random seed
    pre_saving_message = "/home/zhuwu/PUB/RR-Former/runs/4huc_test_decoder_new/Transformer_NAR_[64-4-4-256-0.1]@daymet_4huc_241basins_list_1980~1995#1995~2000#2000~2014@15|14+1[32+1]@NSELoss_n200_bs512_lr0.001_warm_up@seed1234"  # TODO
    used_model = "Transformer"  # TODO

    used_model_config = importlib.import_module(f"configs.model_config.{used_model}_config")
    used_ModelConfig = getattr(used_model_config, f"{used_model}Config")
    decode_mode = used_ModelConfig.decode_mode

    exps_config = list()
    for basin in DatasetConfig.pub_test_basins_list:
        exp_config = dict()
        exp_config["tag"] = basin
        exp_config["ft_train_config"] = {
            "camels_root": DatasetConfig.camels_root,
            "basins_list": [basin],
            "forcing_type": DatasetConfig.forcing_type,
            "start_date": DatasetConfig.train_start,
            "end_date": DatasetConfig.train_end,
            "use_runoff": DataShapeConfig.use_runoff,
            "final_data_path": None
        }

        exp_config["ft_val_config"] = {
            "camels_root": DatasetConfig.camels_root,
            "basins_list": [basin],
            "forcing_type": DatasetConfig.forcing_type,
            "start_date": DatasetConfig.val_start,
            "end_date": DatasetConfig.val_end,
            "use_runoff": DataShapeConfig.use_runoff,
            "final_data_path": None
        }

        exp_config["ft_test_config"] = {
            "camels_root": DatasetConfig.camels_root,
            "basins_list": [basin],
            "forcing_type": DatasetConfig.forcing_type,
            "start_date": DatasetConfig.test_start,
            "end_date": DatasetConfig.test_end,
            "use_runoff": DataShapeConfig.use_runoff,
            "final_data_path": None
        }
        exps_config.append(exp_config)

    if pre_saving_message != "":
        pre_saving_root = ProjectConfig.run_root / pre_saving_message
        pre_model_path = list(pre_saving_root.glob(f"(max_nse)*.pkl"))
        assert (len(pre_model_path) == 1)
        pre_model_path = pre_model_path[0]
        fine_tune_root = pre_saving_root / "fine_tune"


class ValConfig(PretrainLearningConfig):
    seed = 1234  # Random seed
    pre_saving_message = ""  # TODO
    used_model = "Transformer"  # TODO

    used_model_config = importlib.import_module(f"configs.model_config.{used_model}_config")
    used_ModelConfig = getattr(used_model_config, f"{used_model}Config")
    decode_mode = used_ModelConfig.decode_mode

    exps_config = list()
    for basin in DatasetConfig.global_basins_list:
        exp_config = dict()
        exp_config["tag"] = basin
        exp_config["ft_train_config"] = {
            "camels_root": DatasetConfig.camels_root,
            "basins_list": [basin],
            "forcing_type": DatasetConfig.forcing_type,
            "start_date": DatasetConfig.train_start,
            "end_date": DatasetConfig.train_end,
            "use_runoff": DataShapeConfig.use_runoff,
            "final_data_path": None
        }

        exp_config["ft_val_config"] = {
            "camels_root": DatasetConfig.camels_root,
            "basins_list": [basin],
            "forcing_type": DatasetConfig.forcing_type,
            "start_date": DatasetConfig.val_start,
            "end_date": DatasetConfig.val_end,
            "use_runoff": DataShapeConfig.use_runoff,
            "final_data_path": None
        }

        exp_config["ft_test_config"] = {
            "camels_root": DatasetConfig.camels_root,
            "basins_list": [basin],
            "forcing_type": DatasetConfig.forcing_type,
            "start_date": DatasetConfig.test_start,
            "end_date": DatasetConfig.test_end,
            "use_runoff": DataShapeConfig.use_runoff,
            "final_data_path": None
        }
        exps_config.append(exp_config)

    if pre_saving_message != "":
        pre_saving_root = ProjectConfig.run_root / pre_saving_message
        pre_model_path = list(pre_saving_root.glob(f"(max_nse)*.pkl"))
        assert (len(pre_model_path) == 1)
        pre_model_path = pre_model_path[0]
        fine_tune_root = pre_saving_root / "fine_tune"
