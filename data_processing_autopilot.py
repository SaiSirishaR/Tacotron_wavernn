# coding: utf-8

"""
usage: python data_processing_autopilot.py --wav_path <path to wave files> --text_path <path to text files> --feat_path <path to store the extracted features>

"""

import os
import numpy as np
import hparams as hp
from dsp import *
import pickle
import argparse


parser = argparse.ArgumentParser(description='Preprocessing for WaveRNN and Tacotron')
parser.add_argument('--wav_path', '-wp', help='directly point to dataset path (overrides hparams.wav_path')
parser.add_argument('--text_path', '-tp', help='directly point to text')
parser.add_argument('--feat_path', '-fp', help='path to store features')
args = parser.parse_args()

wav_path = args.wav_path
feat_path = args.feat_path
text_path = args.text_path

######## modules to process wavefiles #############

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


def process_wav(wavefile):
    wav_id = wavefile.split('.wav')[0]
    m, x = convert_file(wavefile)
    np.save(feat_path + 'mol/' + wav_id+ '.npy', m, allow_pickle=False)
    np.save(feat_path + 'quant/' + wav_id + '.npy', x, allow_pickle=False)
    return wav_id, m.shape[-1]

###################################################


######## modules to process text #############

def process_text(text_path):
           
                  text_dict ={}
                  print("text_path is", text_path)
                  for file in os.listdir(os.chdir(text_path)):
                      print("file is", file)
                      if file.endswith('.data'):
                         f = open(text_path + '/' + file)
                  
                         for line in f:
                            # split = line.split('|')
                             split = line.split('/n')
                             print("split is", split[-1])
                             text_dict[split[0]] = split[-1]                     
                  return text_dict

###################################################



#### Load data ##########


wav_files = sorted(os.listdir(os.chdir(wav_path)))

dataset = []

for file in wav_files:
    print("processing file...", file)
    processed_id, length = process_wav(file)
    dataset += [(processed_id, length)]
###    print("dataset is", dataset)

###########################


##### dump the wavefile name and lengths #######

with open(feat_path + 'data/' + 'dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)




processed_text = process_text(text_path)


###################################################


##### dump the processed text #######

with open(feat_path + 'data/' + 'text_dict.pkl', 'wb') as f:
            pickle.dump(processed_text, f)



###################################################
