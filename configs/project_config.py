import torch
from pathlib import Path
from configs.dataset_config import DatasetConfig


# Project root, computing resources
class ProjectConfig:
    project_root = Path(__file__).absolute().parent.parent
    single_gpu = 1  # TODO: which gpu to run
    device = torch.device(f"cuda:{single_gpu}")
    # device = torch.device(f"cpu")
    torch.cuda.set_device(device)
    num_workers = 0  # Number of threads for loading data

    # save_dir = f"{DatasetConfig.huc}"
    # save_dir = f"{DatasetConfig.huc}" + "_test_search"
    save_dir = f"{DatasetConfig.huc}" + "_test_decoder_new"
    # save_dir = f"{DatasetConfig.huc}" + "_test_rrs_norunoff_new" #NOTE：
    # save_dir = f"{DatasetConfig.huc}" + "_test_upper"
    # save_dir = f"{DatasetConfig.huc}" + "_test_old"
    # save_dir = f"{DatasetConfig.huc}" + "_test_norunoff"
    # save_dir = f"{DatasetConfig.huc}" + "_test_norunoff_kfold"
    # save_dir = f"{DatasetConfig.huc}" + "_test_norunoff_kfold_trick"
    # save_dir = f"{DatasetConfig.huc}" + "_test_kfold_new"
    run_root = Path(f"./runs/{save_dir}")  # Save each run CHANGE:pub修改了 ORIGIN:Path(f"./runs")

    final_data_root = Path("./final_data")  # Cache preprocessed data NOTE:ORIGIN
    # final_data_root = Path("./final_data_decoder_new")  # Cache preprocessed dat
    # final_data_root = Path("./final_data" + "_test_norunoff")  # Cache preprocessed data
    # final_data_root = Path("./final_data" + "_test_norunoff_kfold")  # Cache preprocessed data
    # final_data_root = Path("./final_data" + "_test_norunoff_kfold_trick")  # Cache preprocessed data
    # final_data_root = Path("./final_data" + "_test_kfold_new")  # Cache preprocessed data
    final_data_root = Path("./final_data" + "_test_decoder_new")  # Cache preprocessed data
    # final_data_root = Path("./final_data" + "_test_rrs_norunoff_new") #NOTE：
