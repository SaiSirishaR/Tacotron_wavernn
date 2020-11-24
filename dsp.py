import hparams as hp
import librosa
import numpy as np

##### Type: Acquisition_CodeBorrowed Source: https://github.com/fatchord/WaveRNN ############

def label_2_float(x, bits):
    return 2 * x / (2**bits - 1.) - 1.

def float_2_label(x, bits):
    assert abs(x).max() <= 1.0
    x = (x + 1.) * (2**bits - 1) / 2
    return x.clip(0, 2**bits - 1)

def normalize(S):
    return np.clip((S - hp.min_level_db) / -hp.min_level_db, 0, 1)


def linear_to_mel(spectrogram):
    return librosa.feature.melspectrogram(
        S=spectrogram, sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels, fmin=hp.fmin)

def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def load_wav(path):
    return librosa.load(path, sr=hp.sample_rate)[0]

def stft(y):
    return librosa.stft(
        y=y,
        n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)

def melspectrogram(y):
    D = stft(y)
    S = amp_to_db(linear_to_mel(np.abs(D)))
    return normalize(S)

