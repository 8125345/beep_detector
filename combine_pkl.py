import pickle
import glob
import os
import math

from scipy.signal import find_peaks

base_path = '/deepiano_data/zhaoliang/SC55_data/iphonexs111_pro_pkl'
# pkl_file_list = glob.glob(os.path.join(base_path, '*'))
# total_pro_list = []
# for pkl_file in sorted(pkl_file_list):
#     with open(pkl_file, 'rb') as f:
#         data = pickle.load(f)
#         print(type(data))
#         total_pro_list = total_pro_list + data
# with open(f'{base_path}/pro_list_total.pkl', "wb") as f_:
#     pickle.dump(total_pro_list, f_)

total_pkl = f'{base_path}/pro_list_total.pkl'
with open(total_pkl, 'rb') as f2:
    data2 = pickle.load(f2)
    print(len(data2))
#
# duration_pkl = '/deepiano_data/zhaoliang/project/spliter_detector/audio_duration.pkl'
# with open(duration_pkl, 'rb') as f1:
#     data1 = pickle.load(f1)
#     print(len(data1))

# peaks, _ = find_peaks(data2, height=0.2)
# print(len(peaks))
# print(peaks)
#
# split_point = []
# time_list = []
# for duration in data1:
#     time = list(duration.values())[0]
#     time_list.append(time)
#
# split_time = 0
# for time_ in time_list:
#     split_time = split_time + time_ + 1.5
#     frame_num = math.ceil(split_time/0.032)
#     split_point.append(frame_num)
# print(split_point)
#
# for split_p in split_point:
#     windows = 100