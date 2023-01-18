import pandas as pd
from pathlib import Path


class DatasetConfig:
    camels_root = Path("/home/zhuwu/Dataset/CAMELS/")  # your CAMELS dataset root
    forcing_type = "daymet"  # TODO: "daymet" or "maurer_extended" or "nldas_extended"
    # forcing_type = "nldas_extended"  # TODO: "daymet" or "maurer_extended" or "nldas_extended"

    huc = "4huc"  # NOTE:4huc[01,03,11,17]
    # basins_test_mark = "01"  # NOTE:for new gauged
    # basin_mark = "t" + basins_test_mark  # NOTE:for new gauged

    basin_mark = "241basins_list"  # NOTE:for kfold or others
    basins_test_mark = "17"  # NOTE:for kfold or others

    basins_file = f"data/{huc}/{basin_mark}.txt"  #
    global_basins_list = pd.read_csv(basins_file, header=None, dtype=str)[0].values.tolist()
    pub_test_basins_list = pd.read_csv(f"data/{huc}/{basins_test_mark}.txt", header=None, dtype=str)[
        0].values.tolist()  # ADD

    # TODO: daymet date
    train_start = pd.to_datetime("1980-10-01", format="%Y-%m-%d")
    train_end = pd.to_datetime("1995-09-30", format="%Y-%m-%d")
    val_start = pd.to_datetime("1995-10-01", format="%Y-%m-%d")
    val_end = pd.to_datetime("2000-09-30", format="%Y-%m-%d")
    test_start = pd.to_datetime("2000-10-01", format="%Y-%m-%d")
    test_end = pd.to_datetime("2014-09-30", format="%Y-%m-%d")

    # TODO: maurer_extended date
    # train_start = pd.to_datetime("2001-10-01", format="%Y-%m-%d")
    # train_end = pd.to_datetime("2008-09-30", format="%Y-%m-%d")
    # val_start = pd.to_datetime("1999-10-01", format="%Y-%m-%d")
    # val_end = pd.to_datetime("2001-09-30", format="%Y-%m-%d")
    # test_start = pd.to_datetime("1989-10-01", format="%Y-%m-%d")
    # test_end = pd.to_datetime("1999-09-30", format="%Y-%m-%d")

    #TODO:KFOLD date
    # train_start = pd.to_datetime("1980-10-01", format="%Y-%m-%d")
    # train_end = pd.to_datetime("1995-09-30", format="%Y-%m-%d")
    # val_start = pd.to_datetime("2010-10-01", format="%Y-%m-%d")
    # val_end = pd.to_datetime("2014-09-30", format="%Y-%m-%d")
    # test_start = pd.to_datetime("1995-10-01", format="%Y-%m-%d")
    # test_end = pd.to_datetime("2010-09-30", format="%Y-%m-%d")


    # dataset_info = f"{forcing_type}{basin_mark}_{train_start.year}~{train_end.year}#{val_start.year}~{val_end.year}#{test_start.year}~{test_end.year}"
    dataset_info = f"{forcing_type}_{huc}_{basin_mark}_{train_start.year}~{train_end.year}#{val_start.year}~{val_end.year}#{test_start.year}~{test_end.year}"
