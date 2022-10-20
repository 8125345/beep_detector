# 数据集整体数量控制，充分利用被下采样的数据集
mul_times = 1
# export PYTHONPATH=$PYTHONPATH:/data/projects/LabelModels
# # export PYTHONPATH=$PYTHONPATH:/data/projects/LabelModels/label_refiner
# export PYTHONPATH=$PYTHONPATH:/data/projects/LabelModels/label_refiner/models

# /Users/xyz/PycharmProjects/XYZ_projects/LabelModels/spliter_detector
# 训练时使用的数据集
train_dataset = [

    ("beep_data_0", 1 * mul_times),
    ("beep_data_1", 1 * mul_times),
    ("beep_data_2", 1 * mul_times),
    ("beep_data_3", 1 * mul_times),

]
# cp forceconcat_20220921_0_default.tflite  forceconcat_20220921_1_default.tflite forceconcat_20220921_2_default.tflite /data/projects/test_models

# 基础配置，如果train_config中存在同名变量，会用train_config变量覆盖basic_config配置
basic_config = {
    "model_root": "/deepiano_data/zhaoliang/project/spliter_detector/train_output_models",
    "train_log_root": "/deepiano_data/zhaoliang/project/spliter_detector/train_log",
    "train_comment_root": "/deepiano_data/zhaoliang/project/spliter_detector/train_comment",
}

# ==================================================================================
# ==================================================================================

# 本次训练配置
# 2022/10/08
train_config = {
    "model_name": f"light_crnn_20221008_0",
    "gpuids": [0],
    "train_batchsize": 128,  # 20是极限
    "val_batchsize": 64,
    "add_maestro": True,
    "pos_weight": 1,
    "pretrain_model": "/deepiano_data/zhaoliang/project/spliter_detector/train_output_models/light_crnn_20220923_0.h5",
    # light_crnn
    "model_structure": "light_crnn",
    "lr": 0.001 / (100),  # 初始学习率
    "dataset_path": "/deepiano_data/zhaoliang/project/spliter_detector/train_data/dataset_beep_20220922_0_train",
    "rec_loss_fun": "weighted_bce",
    "comment": "beep检测",
}
# train_config = {
#     "model_name": f"light_crnn_20220922_0",
#     "gpuids": [0],
#     "train_batchsize": 128,  # 20是极限
#     "val_batchsize": 64,
#     "add_maestro": True,
#     "pos_weight": 1,
#     "pretrain_model": "/data/projects/LabelModels/spliter_detector/train_output_models/light_crnn_20220922_0.h5",
#     # light_crnn
#     "model_structure": "light_crnn",
#     "lr": 0.001 / (10),  # 初始学习率
#     "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_beep_20220922_0_train",
#     "rec_loss_fun": "weighted_bce",
#     "comment": "beep检测",
# }
