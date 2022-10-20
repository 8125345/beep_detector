import os
from shutil import copyfile
import glob

thresholds = 0.5


def read_play_list(m3u_dir):
    assert os.path.isfile(m3u_dir)
    vaild_music_list = []
    with open(m3u_dir, "r", encoding='utf-8') as out:
        music_list = out.readlines()
    music_list_copy = music_list.copy()

    for music in music_list_copy:
        if 'beep' in music:
            music_list_copy.remove(music)
    for mus in music_list_copy:
        music_name = mus.strip().split('/')[-1]
        vaild_music_list.append(music_name)
        # print(music_name)
    return vaild_music_list


if __name__ == '__main__':
    base_path = f'/deepiano_data/zhaoliang/SC55_data/split_result/split_result_{thresholds}_copy'
    m3u_dir = './out.m3u'
    vaild_music_list = read_play_list(m3u_dir)
    print(len(vaild_music_list))

    dir_list = glob.glob(os.path.join(base_path, '*'))
    for dir_ in sorted(dir_list):
        if os.path.isfile(dir_):
            continue
        print(f'当前正在处理{dir_}')
        wav_l = sorted(glob.glob(os.path.join(dir_, '*/*.wav')))
        for i in range(len(vaild_music_list)):
            print(f'正在重命名第{i}个文件')
            _, file = os.path.split(wav_l[i])
            os.rename(wav_l[i], wav_l[i].replace(file, vaild_music_list[i]))










    base_path = f'/deepiano_data/zhaoliang/SC55_data/split_result/split_result_{thresholds}'
