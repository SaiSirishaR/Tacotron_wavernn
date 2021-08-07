# Tacotron_wavernn

Here, I train a multi-speaker TTS. I use WaveRNN vocoder for speech generation.

The following packages are required to run this code snippet.

librosa.
pytorch.
numpy.

**August 7, 2021**

1. Modify the data split line in the function process_text

   split = line.split('|') --> for LJspeech.     
   split = line.split('/n') --> for arctic.

2. modify the sampling rate in hparams file (16kHz for arctic and 22050 for LJspeech)

3. python data_processing_autopilot.py --wav_path <path to wave files> --text_path <path to text files> --feat_path <path to store the extracted features>
