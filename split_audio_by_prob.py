import pickle
# import scipy
from scipy.signal import find_peaks
import numpy as np

# 加载切分区间
audio_dur_path = "/deepiano_data/zhaoliang/project/spliter_detector/audio_duration.pkl"
with open(audio_dur_path, "rb") as f:
    audio_dur = pickle.load(f)
audio_num = len(audio_dur)
print(f"音频数量：{audio_num}")  # 音频数量：400
audio_dur_list = list()
for d in audio_dur:
    dv = None
    for k, v in d.items():
        dv = v
    audio_dur_list.append(dv)

# 预测结果列表
process_list = [
    "/deepiano_data/zhaoliang/SC55_data/iphonexs111_pro_pkl/pro_list_total.pkl",
]





# 对于每个大的预测结果做切分点估计
for path in process_list:
    # 加载beep概率结果
    with open(path, "rb") as f:
        prob_data = pickle.load(f)

    prob_data = np.array(prob_data)

    # todo 格式转换
    threshold = 0.01
    peaks, peak_heights = find_peaks(prob_data, height=threshold)
    peak_heights = peak_heights["peak_heights"]
    print(f"文件：{path}\t\t检测出峰值的数量为：{len(peaks)}")
    first_split_idx = np.where(prob_data > 0.5)[0][0]  # 获得第一个元素序号
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
    print(split_list)
    print(len(split_list))




