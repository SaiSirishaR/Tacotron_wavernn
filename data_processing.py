import os
import numpy as np
import hparams as hp
from dsp import *
import pickle


######## modules to process wavefiles #############

# Type: Acquisition_CodeBorrowed Source: https://github.com/fatchord/WaveRNN 

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


def process_wav(data_dir, wavefile):
    wav_id = wavefile.split('.wav')[0]
    m, x = convert_file(wavefile)
    np.save(data_dir + 'mol/' + wav_id+ '.npy', m, allow_pickle=False)
    np.save(data_dir + 'quant/' + wav_id + '.npy', x, allow_pickle=False)
    return wav_id, m.shape[-1]

###################################################


######## modules to process text #############

def process_text(text_path, data_dir):
           
           text_dict ={}

           for file in data_dir:
               print("file is", file)
               if file.endswith('.csv'):
                  f = open(text_path + '/' + file)
                  for line in f:
                      split = line.split('|')
                      print("split is", split[-1])
                      text_dict[split[0]] = split[-1]                     
       #    print("dict is", text_dict)
           return text_dict

###################################################


#### Load data ##########

data_path = 'path/to/data directory'
wave_path = data_path + 'wavs/'
wav_files = sorted(os.listdir(os.chdir(wave_path)))

dataset = []

for file in wav_files:
    print("processing file...", file)
    processed_id, length = process_wav(data_path, file)
    dataset += [(processed_id, length)]
###    print("dataset is", dataset)

###########################


##### dump the wavefile name and lengths #######

with open(data_path + 'data/' + 'dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)

processed_text = process_text(data_path, os.listdir(data_path))

###################################################


##### dump the processed text #######

with open(data_path + 'data/' + 'text_dict.pkl', 'wb') as f:
            pickle.dump(processed_text, f)

###################################################
