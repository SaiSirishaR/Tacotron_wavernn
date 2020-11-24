import os
import numpy as np
import hparams as hp
from dsp import *

##### Type: Acquisition_CodeBorrowed Source: https://github.com/fatchord/WaveRNN ############

def convert_file(wave):
    y = load_wav(wave)
    peak = np.abs(y).max()
    if hp.peak_norm or peak > 1.0:
        y /= peak
    mel = melspectrogram(y)
    if hp.voc_mode == 'RAW':
        quant = encode_mu_law(y, mu=2**hp.bits) if hp.mu_law else float_2_label(y, bits=hp.bits)
    elif hp.voc_mode == 'MOL':
        quant = float_2_label(y, bits=16)

    return mel.astype(np.float32), quant.astype(np.int64)

#########################################################################################################

##### Type: Acquisition_CodeBorrowed Source: https://github.com/fatchord/WaveRNN ############


def process_wav(data_dir, wavefile):
    wav_id = wavefile.split('.wav')[0]
    m, x = convert_file(wavefile)
    np.save(data_dir + 'mol/' + wav_id+ '.npy', m, allow_pickle=False)
    #np.save(paths.quant/f'{wav_id}.npy', x, allow_pickle=False)
    return wav_id, m.shape[-1]
#########################################################################################################


data_path = '/Users/sirisha/Documents/Projects/PhD_Projects/Interspeech2021/Project_TWlstm_vs_fb/Data/LJspeech/'
wave_path = data_path + 'LJSpeech-1.1/wavs/'
wav_files = sorted(os.listdir(os.chdir(wave_path)))


for file in wav_files:
    print("processing file...", file)
    processed, length = process_wav(data_path, file)

