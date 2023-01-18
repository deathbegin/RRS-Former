# Define the shape of data
class DataShapeConfig:
    past_len = 14  # Length of formerly known runoff sequence NOTE:15
    pred_len = 1  # Length of runoff sequence to be predicted NOTE:7
    use_future_fea = True  # Whether to use future meteorological data, usually True here
    if use_future_fea:
        src_len = past_len + pred_len  # 15 + 7
    else:
        src_len = past_len  # 15
    tgt_len = past_len + pred_len  # 15 + 7
    ##########################################################################
    dynamic_size = 5  # Number of dynamic attributes
    static_size = 27  # Number of static attributes

    use_runoff = False  # TEST
    if use_runoff is True:
        dynamic_size += 1  # NOTE:dynamic_size=6

    use_static = True  # TODO: whether to use static feature
    if use_static is True:
        src_size = dynamic_size + static_size  # 5 + 27
    else:
        src_size = dynamic_size  # 27

    tgt_size = 1  # Number of target attributes, always 1 here since we only need to predict runoff

    data_shape_info = f"{src_len}|{past_len}+{pred_len}[{src_size}+{tgt_size}]"
