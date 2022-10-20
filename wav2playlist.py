import os
import glob
import subprocess
from shutil import copyfile


if __name__ == '__main__':
    SC55_wav_dir = '/Users/xyz/Downloads/xml_wav/xml_SC55'
    arachno_wav_dir = '/Users/xyz/Downloads/xml_wav/xml_arachno'
    beep = '/Users/xyz/Desktop/beep/beep.wav'
    out_m3u = '/Users/xyz/Desktop/beep/out.m3u'
    out_dir = '/Users/xyz/Desktop/SC55'

    SC55_wav_files = glob.glob(os.path.join(SC55_wav_dir, '*.wav'))
    arachno_wav_files = glob.glob(os.path.join(arachno_wav_dir, '*.wav'))
    SC55_wav_files.sort()
    arachno_wav_files.sort()
    result = []
    result_beep = []
    result_SC55 = []
    result_arachno = []
    for i in range(0, 400):
        i = str(i).zfill(3)
        out_temp_beep = os.path.join(out_dir, f'beep_{i}.wav')
        # copyfile(beep, out_temp_beep)
        result_beep.append(out_temp_beep)


    for SC55 in SC55_wav_files:
        parent = SC55.split('/')[-2]
        filename = SC55.split('/')[-1]
        SC55_out_temp_dir = os.path.join(out_dir, f'{parent}_{filename}')
        # copyfile(SC55, SC55_out_temp_dir)
        result_SC55.append(SC55_out_temp_dir)
    for arachno in arachno_wav_files:
        parent = arachno.split('/')[-2]
        filename = arachno.split('/')[-1]
        arachno_out_temp_dir = os.path.join(out_dir, f'{parent}_{filename}')
        # copyfile(arachno, arachno_out_temp_dir)
        result_arachno.append(arachno_out_temp_dir)
    j = 0
    for i, beep in enumerate(result_beep):
        if i < 200:
            result.append(result_SC55[i])
            result.append(beep)
        elif i > 199:
            result.append(result_arachno[j])
            result.append(beep)
            j = j + 1

    with open(out_m3u, "w", encoding='utf-8') as out:
        for res in result:
            out.write(str(res) + "\n")




