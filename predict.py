import os
import glob
import math
from pkl_to_midi import pkl_to_mid
from shutil import copyfile
from scipy.signal import find_peaks
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 推理时限定GPU

import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

batch_size = 1  # 尽量吧GPU打满
thresholds = 0.5

"""
# # keras计算
# # precision
# m = tf.keras.metrics.Precision()
# m.update_state(y_predict, y_label)
# precision = m.result().numpy()
# print(f"precision: {precision}")
#
# # recall
# m.reset_state()
# m = tf.keras.metrics.Recall()
# m.update_state(y_predict, y_label)
# recall = m.result().numpy()
# print(f"recall: {recall}")
"""


def parse_json_and_serialized_data(data:dict):
    cnt = 0
    feature_list = list()
    label_list = list()
    for _, path in data.items():
        cnt += 1
        serialized = tf.io.read_file(path)
        data = tf.io.parse_tensor(serialized, tf.float32)
        feature = data[:, : 293120]
        midi = data[:, 293120:]
        feature = tf.reshape(feature, (640, 229, 2))
        feature = feature[:, :, 1]  # 获取mix
        midi = tf.reshape(midi, (640, 1))

        feature = feature.numpy()
        midi = midi.numpy()

        feature_list.append(feature)
        label_list.append(midi)

    feature_arr = np.array(feature_list)
    label_arr = np.array(label_list)
    return feature_arr, label_arr


def predict_feature(model, feature_arr, label_arr):
    rst = model.predict(feature_arr, batch_size=batch_size)
    print(rst.shape)
    # 计算评估指标
    thresholds = 0.5
    y_label = label_arr.reshape(-1)
    y_predict = rst.reshape(-1) > thresholds

    test_arr = np.vstack((y_label, y_predict))

    # sklearn计算
    pre, rec, f1, sup = precision_recall_fscore_support(y_label, y_predict, average="binary", pos_label=1)
    # print(pre, rec, f1)
    return pre, rec, f1


def get_all_predict_data(json_dir):
    json_file_list = []
    if os.path.isdir(json_dir):
        json_file_list = glob.glob(os.path.join(json_dir,'*.json'))
        print('json数量：', len(json_file_list))
    return json_file_list


def Gather_json(json_file_list):
    gather = dict()
    path_list = list()
    for json_file in json_file_list:
        with open(json_file, "r") as f:
            data = json.load(f)
            for _, path in data.items():
                path_list.append(path)
    print('序列化样本数量：', len(path_list))
    for i, path in enumerate(sorted(path_list)):
        gather[i] = path
    return gather


def test_one_data(json_file_list):
    test_json = json_file_list[0]
    test_list = list()
    test_sample = dict()

    cnt = 0
    with open(test_json, "r") as f:
        data = json.load(f)
        for _, path in data.items():
            test_list.append(path)
            cnt += 1
            if cnt == 3:
                break
    for i, path in enumerate(test_list):
        test_sample[i] = path
    # print(test_sample)
    return test_sample


def test_one(json_file_list):
    test_sample_one = test_one_data(json_file_list)
    print(test_sample_one)
    feature_arr, label_arr = parse_json_and_serialized_data(test_sample_one)
    rst = model.predict(feature_arr, batch_size=1)
    print(rst.shape)
    sample_num = rst.shape[0]
    thresholds = 0.5
    y_label = label_arr.reshape(-1)
    y_predict = rst.reshape(-1) > thresholds
    print('预测', y_predict.shape)
    # print('标注：', y_label)
    y_predict = y_predict.reshape(sample_num, -1)

    print('预测', y_predict.shape)
    y_predict = y_predict.astype(int)
    # print(y_predict)


    index_1 = np.argwhere(y_predict == 1)
    # index_1 = index_1.reshape(-1)
    # print(y_predict)
    print(index_1)


def generate_anytime_spec():#测试可以切分一个半小时的音频
    test_duration = 5400  #s
    duration_per_frame = 0.032
    frame_num = int(math.ceil(test_duration / duration_per_frame))
    print('帧数为：', frame_num)
    sepc = np.ones([1, frame_num, 229])
    print(sepc.shape)

    return sepc


def parse_np_data(path):
    data = np.load(path)
    chunk_spec = data[:, : 229]
    bgm_chunk_spec = data[:, 229: 2 * 229]
    onsets_label = data[:, 2 * 229:]
    return bgm_chunk_spec, chunk_spec, onsets_label


def get_all_npy_data(song_folder_path):
    assert os.path.exists(song_folder_path)
    file_paths = glob.glob(os.path.join(song_folder_path, "*.npy"))

    def sort_fun(path):
        f_name = os.path.split(path)[-1]
        return int(os.path.splitext(f_name)[0])

    file_paths = sorted(file_paths, key=sort_fun)
    return file_paths
    # for file_path in file_paths:
    #     bgm, mix, midi = parse_np_data(file_path)


def copyfile2file(npy_file):
    wav_ori_dir = npy_file.replace('beep_predict_result/beep_predict_result_0.3', 'npy_data').replace('.npy', '.wav')
    assert os.path.isfile(wav_ori_dir)
    wav_dst_dir = npy_file.replace('.npy', '.wav')
    copyfile(wav_ori_dir, wav_dst_dir)


def combine_pkl(input_dir):
    pkl_file_list = glob.glob(os.path.join(input_dir, '*.pkl'))
    total_pro_list = []
    for pkl_file in sorted(pkl_file_list):
        with open(pkl_file, 'rb') as f1:
            data = pickle.load(f1)
            total_pro_list = total_pro_list + data
    with open(f'{input_dir}/pro_list_total.pkl', "wb") as f2:
        pickle.dump(total_pro_list, f2)


if __name__ == '__main__':
    # json_dir = '/deepiano_data/zhaoliang/SplitModel_data/json_with_beep'
    input_dir = '/deepiano_data/zhaoliang/SC55_data/npy_data'
    model_path = "/data/projects/LabelModels/spliter_detector/train_output_models/light_crnn_20220923_0.h5"
    out_dir = f'/deepiano_data/zhaoliang/SC55_data/beep_predict_result/beep_predict_result_{thresholds}'
    model = load_model(model_path, compile=False)

    # json_file_list = get_all_predict_data(json_dir)

    # gather = Gather_json(json_file_list)
    # feature_arr, label_arr = parse_json_and_serialized_data(gather)
    # pre, rec, f1 = predict_feature(model, feature_arr, label_arr)
    # print('该数据集上精确度：', pre)
    # print('该数据集上召回率：', rec)
    # print('该数据集上F1：', f1)
    #测试预测一个样本
    # sepc_res = generate_anytime_spec()

    song_list = glob.glob(os.path.join(input_dir, '*'))
    for song_id in sorted(song_list):
        song_name = song_id.split('/')[-1]
        out_dir_file = os.path.join(out_dir, song_name)
        if not os.path.isdir(out_dir_file):
            os.makedirs(out_dir_file)
        npy_list = get_all_npy_data(song_id)
        total_pro_list = []
        for i, npy_file in enumerate(npy_list):
            _, f = os.path.split(npy_file)
            fn, _ = os.path.splitext(f)
            dst_dir_npy = os.path.join(out_dir_file, f'{fn}.npy')
            print(dst_dir_npy)

            _, mix, _ = parse_np_data(npy_file)
            mix = mix.reshape(1, mix.shape[0], mix.shape[1])
            rst = model.predict(mix, batch_size=batch_size)
            print(rst.shape)
            predict_result = rst.reshape(-1)
            predict_result_list = predict_result.tolist()

            predict_result = rst.reshape(-1) > thresholds
            predict_result = np.squeeze(predict_result, axis=None)
            print(predict_result.shape)
            predict_result = predict_result.astype(int)
            np.save(dst_dir_npy, predict_result)
            with open(f'{out_dir_file}/pro_list_{i}.pkl', "wb") as f:
                pickle.dump(predict_result_list, f)
        combine_pkl(out_dir_file)












