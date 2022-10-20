import glob
import numpy as np
import os
import pandas as pd
import pickle

thresholds = 0.5

pd.set_option("display.max_rows", None,
              "display.max_columns", None,
              "display.max_colwidth", 100,
              "display.width", 100)

from deepiano.wav2mid.audio_transform import get_audio_duration


def get_all_split_wav_data(song_folder_path):
    assert os.path.exists(song_folder_path)
    file_paths = glob.glob(os.path.join(song_folder_path, "*.wav"))
    def sort_fun(path):
        f_name = os.path.split(path)[-1]
        return int(os.path.splitext(f_name)[0])

    file_paths = sorted(file_paths, key=sort_fun)
    return file_paths


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


def get_audio_duration_dict(wav_dir):
    audio_duration_dict = dict()
    xml_name_list = glob.glob(os.path.join(wav_dir, '*'))
    for xml_dir in sorted(xml_name_list):
        xml_name = xml_dir.split('/')[-1]
        song_list = get_all_split_wav_data(xml_dir)
        for song_id in song_list:
            fn = song_id.split('/')[-1]
            dst_file = xml_name + '_' + fn
            song_id_duration = get_audio_duration(song_id)
            audio_duration_dict[dst_file] = song_id_duration

    return audio_duration_dict


def get_audio_duration_list(wav_dir):
    audio_duration_list = []
    xml_name_list = glob.glob(os.path.join(wav_dir, '*'))
    for xml_dir in sorted(xml_name_list):
        xml_name = xml_dir.split('/')[-1]
        song_list = get_all_split_wav_data(xml_dir)
        for song_id in song_list:
            song_id_dict = dict()
            fn = song_id.split('/')[-1]
            dst_file = xml_name + '_' + fn
            song_id_duration = get_audio_duration(song_id)
            song_id_dict[dst_file] = song_id_duration
            audio_duration_list.append(song_id_dict)

    return audio_duration_list


def get_record_audio_duration_dict(split_result_dir):
    record_audio_duration_dict_total = dict()
    audio_list = glob.glob(os.path.join(split_result_dir, '*'))
    for audio in sorted(audio_list):
        record_audio_duration_dict = dict()
        audio_name = audio.split('/')[-1]
        split_chunk_1 = os.path.join(audio, 'total')
        # split_chunk_name = split_chunk_1.split('/')[-1]
        split_chunk_2_wav_list = get_all_split_wav_data(split_chunk_1)
        for split_chunk_2_wav in split_chunk_2_wav_list:
            x = split_chunk_2_wav.split('/')
            split_chunk_2_wav_name = x[-3] + '/' + x[-2] + '/' + x[-1]
            record_audio_duration_dict[split_chunk_2_wav_name] = get_audio_duration(split_chunk_2_wav)

        record_audio_duration_dict_total[audio_name] = record_audio_duration_dict

    return record_audio_duration_dict_total


def dict2csv(excel_file, audio_duration_dict, record_audio_duration_dict_total):

    writer = pd.ExcelWriter(excel_file)
    result_1 = pd.DataFrame().from_dict(audio_duration_dict, orient='index', columns=['duration'])
    result_1.to_excel(writer, encoding='utf-8')
    # print(result_1)
    for i, r_a_d_d in enumerate(record_audio_duration_dict_total.items()):
        temp_dict = r_a_d_d[1]
        result_2 = pd.DataFrame().from_dict(temp_dict, orient='index', columns=['duration'])
        i += 1
        StartCol = i * 5
        result_2.to_excel(writer, startcol=StartCol, encoding='utf-8')
        # print(result_2)
    writer.save()


if __name__ == '__main__':
    split_result_dir = f'/deepiano_data/zhaoliang/SC55_data/split_result/split_result_{thresholds}'
    m3u_dir = './out.m3u'
    wav_dir = '/deepiano_data/zhaoliang/xml_wav'
    excel_file = f'/deepiano_data/zhaoliang/SC55_data/split_result/split_result_{thresholds}/duration_match_{thresholds}.xlsx'

    audio_duration_dict = get_audio_duration_dict(wav_dir)
    record_audio_duration_dict_total = get_record_audio_duration_dict(split_result_dir)
    dict2csv(excel_file, audio_duration_dict, record_audio_duration_dict_total)










