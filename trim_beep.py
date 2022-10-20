import glob
import sox
import numpy as np
import os
import subprocess
from pkl_to_midi import pkl_to_mid
from shutil import copyfile
import pickle
from scipy.signal import find_peaks

from deepiano.wav2mid.audio_transform import get_audio_duration

duration_per_frame = 0.032
thresholds = 0.5

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


def npy_to_midi(midi_data, dst_file):
    midi_data = midi_data.reshape(midi_data.shape[0], 1)
    pkl_to_mid.convert_to_midi_single(midi_data, dst_file)


def index_to_time(index_list):
    beep_onset_time_list = []
    for index_ in index_list:
        beep_onset_time = round((index_ - 1) * duration_per_frame, 5)
        beep_onset_time_list.append(beep_onset_time)
    return beep_onset_time_list


def trim_audio_function(input_audio, output_file_str, start_time, input_duration):

    trim_command = 'sox {input_audio} {output_file} trim {start} {input_duration}'.format(**{
        'input_audio': input_audio,
        'output_file': output_file_str,
        'start': start_time,
        'input_duration': input_duration,
    })
    print(trim_command)
    process_handle = subprocess.Popen(trim_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    process_handle.communicate()


def trim_audio_work(beep_onset_time_list, input_audio, split_out_dir):
    audio_id = input_audio.split('/')[-2]
    fn = input_audio.split('/')[-1].split('.')[-2]
    dst_dir = split_out_dir + '/' + audio_id + '/' + fn + '/'
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    if beep_onset_time_list:
        start_time = 0
        cnt = 0
        for beep_onset_time in beep_onset_time_list:
            print(f"第{cnt}次截取的开始时间是", start_time)
            input_duration = round((beep_onset_time - 0.5 - start_time), 5)#onset时间前移500ms
            if input_duration < 0:
                print('持续时长为0，raise an error')
                break
            print(f"第{cnt}次截取的持续时间是", input_duration)
            out_part_file = os.path.join(dst_dir, '%06d.wav' % cnt)
            print(f"第{cnt}次保存的文件是", out_part_file)
            trim_audio_function(input_audio, out_part_file, start_time, input_duration)
            start_time = beep_onset_time + 1.5 #onset时间后移1.5s
            cnt += 1
        print('循环次数', cnt)
        if cnt == len(beep_onset_time_list):
            audio_duration = get_audio_duration(input_audio)
            print(audio_duration)
            print(f"第{cnt}次截取的开始时间是", start_time)
            input_duration = round((audio_duration - start_time), 5)  # 截取最后一段
            print(f"第{cnt}次截取的持续时间是", input_duration)
            out_part_file = os.path.join(dst_dir, '%06d.wav' % cnt)
            print(f"第{cnt}次保存的文件是", out_part_file)
            trim_audio_function(input_audio, out_part_file, start_time, input_duration)


def copyfile2file(npy_file):
    wav_ori_dir = npy_file.replace(f'beep_predict_result/beep_predict_result_{thresholds}', 'npy_data').replace('.npy', '.wav')
    assert os.path.isfile(wav_ori_dir)
    wav_dst_dir = npy_file.replace('.npy', '.wav')
    copyfile(wav_ori_dir, wav_dst_dir)


def correct_list(index_list):
    index_list_correct = index_list.copy()
    for i, index_ in enumerate(index_list_correct):
        if i > 0:
            diff_frame = index_list_correct[i] - index_list_correct[i-1]
            if abs(diff_frame) < 10:
                index_list_correct.remove(index_list_correct[i])
    return index_list_correct


def get_record_audio_duration(audio_dur_path):
    # 加载切分区间
    with open(audio_dur_path, "rb") as f:
        audio_dur = pickle.load(f)
    audio_num = len(audio_dur)
    # print(f"音频数量：{audio_num}")  # 音频数量：400
    audio_dur_list = list()
    for d in audio_dur:
        dv = None
        for k, v in d.items():
            dv = v
        audio_dur_list.append(dv)
    return audio_num, audio_dur_list


def second2frame(time):
    """
    秒时间单位转化为frame单位
    :param time:
    :return:
    """
    frame_num = int(time * 1000 / 32)
    return frame_num


def predict_after_proccess(total_pkl_file, audio_num, audio_dur_list):
    assert os.path.isfile(total_pkl_file)
    with open(total_pkl_file, "rb") as f_pkl:
        prob_data = pickle.load(f_pkl)

    prob_data = np.array(prob_data)

    # todo 格式转换
    threshold = 0.01
    peaks, peak_heights = find_peaks(prob_data, height=threshold)
    peak_heights = peak_heights["peak_heights"]
    print(f"文件：{total_pkl_file}\t\t检测出峰值的数量为：{len(peaks)}")
    first_split_idx = np.where(prob_data > thresholds)[0][0]  # 获得第一个元素序号
    data_len = prob_data.shape[0]

    # 所有合适的切分点
    split_list = list()
    split_list.append(first_split_idx)
    # 遍历所有音频每个音频都要找到一个合适的分割节点


    for i in range(audio_num):
        # 对于第0个歌曲，直接以第一个切分点作为基准
        if i == 0:
            continue
        # 每个切分点的可能区域都是利用上一个切分点位置还有时长作为先验信息
        last_split_idx = split_list[-1]
        # 本次可能的位置为
        noise = second2frame(0.195)

        cur_audio_guess_idx = last_split_idx + second2frame(audio_dur_list[i]) + second2frame(
            2) + noise  # 上一个位置+本次音频长度+split时间
        cur_roi_start = cur_audio_guess_idx - second2frame(1)
        cur_roi_end = cur_audio_guess_idx + second2frame(1)

        # 从开始和结束之间的峰值里面找到数值最大的元素
        roi_peaks = list()
        max_p_value = -1
        max_p_idx_in_roi = -1
        for p_i in range(len(peaks)):
            if cur_roi_start <= peaks[p_i] <= cur_roi_end:
                # 直接找到这段区间的最大值
                if peak_heights[p_i] > max_p_value:
                    max_p_value = peak_heights[p_i]
                    max_p_idx_in_roi = peaks[p_i]  # 保存区间最大峰值索引
                roi_peaks.append(peaks[p_i])
        if max_p_idx_in_roi == -1:
            # raise Exception("未找到切分区间")
            print(f"第{i}段音乐后面未找到合适切分点，使用估计值：{cur_audio_guess_idx}")
            max_p_idx_in_roi = cur_audio_guess_idx
        # print(f"选中位置和估计位置差值：{cur_audio_guess_idx - max_p_idx_in_roi}")  # 测试算法准确性，调试阶段使用
        split_list.append(max_p_idx_in_roi)

    # split_list就是最终的切分点
    # print(split_list)
    print(len(split_list))
    return split_list

def combine_wav(data:list, new_file):
    cbn = sox.Combiner()
    cbn.build(data, new_file, 'concatenate')


if __name__ == '__main__':
    audio_dur_path = "/deepiano_data/zhaoliang/project/spliter_detector/audio_duration.pkl"
    base_path = f'/deepiano_data/zhaoliang/SC55_data/beep_predict_result/beep_predict_result_{thresholds}'
    split_out_dir = f'/deepiano_data/zhaoliang/SC55_data/split_result/split_result_{thresholds}'

    audio_num, audio_dur_list = get_record_audio_duration(audio_dur_path)

    audio_list = glob.glob(os.path.join(base_path, '*'))
    for audio in sorted(audio_list):
        npy_file_list = get_all_npy_data(audio)
        pro_list_total_pkl = os.path.join(audio, 'pro_list_total.pkl')
        assert os.path.isfile(pro_list_total_pkl)
        wav_list = []
        for npy_file in npy_file_list:
            copyfile2file(npy_file)
            midi_dst_file = npy_file.replace('.npy', '.mid')
            data = np.load(npy_file)
            npy_to_midi(data, midi_dst_file)
            # index = np.argwhere(data == 1)
            # index = index.reshape(-1)
            # index_list = index.tolist()
            # index_list_correct = correct_list(index_list)
            audio_chunk = npy_file.replace('.npy', '.wav')
            assert os.path.isfile(audio_chunk)
            wav_list.append(audio_chunk)
        index_list_correct = predict_after_proccess(pro_list_total_pkl, audio_num, audio_dur_list)
        beep_onset_time_list = index_to_time(index_list_correct)
        # print(beep_onset_time_list)
        wav_total_file = os.path.join(audio, 'total.wav')
        combine_wav(sorted(wav_list), wav_total_file)
        assert os.path.isfile(wav_total_file)
        trim_audio_work(beep_onset_time_list, wav_total_file, split_out_dir)











    # test_audio_one = audio_list[2]
    # # print(test_one)
    # npy_file_list = get_all_npy_data(test_audio_one)
    # test_npy_one = npy_file_list[0]
    # # print(test_npy_one)
    # for npy_file in npy_file_list:
    #     data = np.load(test_npy_one)
    #     data = data.reshape(data.shape[0], 1)
    #     pkl_to_mid.convert_to_midi_single(data, f'{out_dir}/000000.mid')
    #
    #     index = np.argwhere(data == 1)
    # print(data.shape)
    # print(len(index))
    # print(data.shape)










