"""
2022.7.22
用于清晨项目，数据切分为大块，加载后内存中切分
新版声音识别数据集
！注意：使用tf2.*环境执行此脚本，tf1.15存在内存泄漏
根据数据集json文件生成tfrecord
"""
import sys

# sys.path.insert(0, '/data/projects/BGMcloak')
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import random
import pickle
import json
from spliter_detector.tfrecord_tools import *

random.seed(666)


def decoder_fun(line):
    """
    直接解析每行数据，创建tfrecord的feature
    :param line: 传入数据
    :return: 分别返回固定长度和非固定长度数据
    """

    path = line
    feature = {
        "path": _bytes_feature(path.encode("utf-8")),
    }

    return feature


def split_dataset(data_list, val_ratio=0.1, shuffle=True):
    """
    训练集、验证集切分甲苯
    :param data_list:
    :param val_ratio:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data_list)
    total_num = len(data_list)
    val_dataset_num = int(total_num * val_ratio)
    train_dataset_num = total_num - val_dataset_num

    train_list = data_list[: train_dataset_num]
    val_list = data_list[train_dataset_num:]
    return train_list, val_list


def divide_list_n(my_list, n):
    tmp_list = [[] for i in range(n)]
    for i, e in enumerate(my_list):
        tmp_list[i % n].append(e)
    return tmp_list


def save_sep_list(root_path, divide_data, data_type):
    assert data_type in ["train", "val"]
    # 保存分片训练数据
    for data_idx, seg_list in enumerate(divide_data):
        print(f"分片{data_idx}，数量{len(seg_list)}")
        with open(os.path.join(root_path, f"seg_{data_type}_list_{data_idx}.pkl"), "wb") as f:
            pickle.dump(seg_list, f)


def load_sep_list(root_path, data_save_path, data_type, total_seg_num):
    assert data_type in ["train", "val"]

    data_num = 0
    # 分片生成tfrecord数据
    for data_idx in range(total_seg_num):
        print(f"生成{data_type}分片{data_idx}")

        with open(os.path.join(root_path, f"seg_{data_type}_list_{data_idx}.pkl"), "rb") as f:
            seg_train_list = pickle.load(f)

        seg_train_num = create_tfrecord(seg_train_list, data_save_path + f"_{data_idx}", decoder_fun)
        data_num += seg_train_num
        print(f"{data_type}数据：{data_num}")
    return data_num


def run_single_schedule(dataset_type, root_path, json_dataset):
    """
    将json数据集
    :param dataset_type: "train" "test"
    :param root_path: 数据保存位置
    :param json_path: json数据集路径
    :return:
    """
    print(f"当前处理：{dataset_type}")
    dataset_name = json_dataset["name"]
    json_path = json_dataset["path"]
    sample_ratio = json_dataset["ratio"]  # 下采样的比例，实际使用数据数量=原始数量*sample_ratio
    assert 0 < sample_ratio <= 1
    if dataset_type == "train":
        # 训练集数据处理
        val_ratio = 0.1
        # train_num: 9283575
        # val_num: 488609
        assert os.path.exists(root_path)
        # 验证集比例

        # json_path = os.path.join(root_path, "dataset.json")
        # json_path = "/deepiano_data/yuxiaofei/work/data_0718/serialize/ai-tagging_train.json"  # todo
        train_save_path = os.path.join(root_path, f"train_{dataset_name}.tfrecord")
        val_save_path = os.path.join(root_path, f"val_{dataset_name}.tfrecord")
        dataset_info = os.path.join(root_path, f"dataset_info_{dataset_name}.json")
        random_state_save_path = os.path.join(root_path, f"random_state_{dataset_name}.pkl")

        # 保存随机状态
        start_state = random.getstate()
        pickle.dump(start_state, open(random_state_save_path, "wb"))

        # 数据生成完成
        print("数据加载中...")
        with open(json_path, "r") as f:
            data = json.load(f)

        # 转为list
        data_list = list()
        for idx, s_path in data.items():
            data_list.append(s_path)

        print("数据切分中...")
        # 片段级shuffle
        train_list, val_list = split_dataset(data_list, val_ratio=val_ratio, shuffle=True)

        print('数据下采样中...')
        train_list = random.sample(train_list, int(len(train_list) * sample_ratio))
        val_list = random.sample(val_list, int(len(val_list) * sample_ratio))

        # train_list离线保存避免内存爆炸
        print("训练tfrecord生成中...")
        train_num = create_tfrecord(train_list, train_save_path, decoder_fun)
        print(f"训练数据：{train_num}")
        print(f"训练数据校验{len(train_list)}")

        print("验证tfrecord生成中...")
        val_num = create_tfrecord(val_list, val_save_path, decoder_fun)
        print(f"验证数据：{val_num}")

        info_dict = {
            "train_num": train_num,
            "val_num": val_num
        }
        with open(dataset_info, "w") as f:
            json.dump(info_dict, f)
        print(f"{json_path}处理完成")


def make_schedule():
    process_list = list()  # 执行计划
    # 控制生成数据集类型
    dataset_types = [
        "train",
        # "test"
    ]

    json_datasets = [

        # 无BGM真实演奏+后处理添加BGM
        {"name": "beep_data_0",
         "path": "/deepiano_data/zhaoliang/SplitModel_data/json_with_beep/beep_train_0.json",
         "ratio": 1},
        {"name": "beep_data_1",
         "path": "/deepiano_data/zhaoliang/SplitModel_data/json_with_beep/beep_train_1.json",
         "ratio": 1},
        {"name": "beep_data_2",
         "path": "/deepiano_data/zhaoliang/SplitModel_data/json_with_beep/beep_train_2.json",
         "ratio": 1},
        {"name": "beep_data_3",
         "path": "/deepiano_data/zhaoliang/SplitModel_data/json_with_beep/beep_train_3.json",
         "ratio": 1},


    ]

    # 每次需要提前手工创建目录，如果自动创建切记不要影响历史目录
    dataset_name = "dataset_beep_20220922_0"  # 去除无BGM演奏，避免召回降低
    # 创建数据根目录
    for dataset_type in dataset_types:
        # 数据保存路径
        root_dst_folder = "/data/dataset/audio/rec_dataset/dataset_info_concat"
        folder_name = f"{dataset_name}_{dataset_type}"  # 新数据集
        dst_folder = os.path.join(root_dst_folder, folder_name)
        assert os.path.exists(dst_folder)
        process_list.append(
            {"dataset_type": dataset_type, "dst_folder": dst_folder, "json_datasets": json_datasets},
        )
    print(process_list)

    return process_list


def run_schedule(schedule):
    # 执行处理计划
    for item in schedule:
        dataset_type = item["dataset_type"]
        root_path = item["dst_folder"]
        json_datasets = item["json_datasets"]

        with open(os.path.join(root_path, "dataset_elements.json"), "w") as f:
            json.dump(json_datasets, f)

        # 分别处理每个数据集
        for json_dataset in json_datasets:
            run_single_schedule(dataset_type, root_path, json_dataset)


def process_dataset():
    # 设计数据集处理计划
    schedule = make_schedule()
    # 执行计划
    run_schedule(schedule)

    print("程序完成")


if __name__ == '__main__':
    process_dataset()
