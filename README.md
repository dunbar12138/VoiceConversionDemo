# VoiceConversionDemo

First download the trained models and save them somewhere.

Next check audio2audio.py and change 'model_path' to where you save those models

How to use it: python audio2audio.py --wav_path inputsample/mysampleinput1.m4a --model obama_neural --parallel

It will take 5-10 seconds to load the model and another few seconds to translate the input. The output will be stored as "result2.wav" in this directory.


## Environment Setup

python = 3.6

pip install -r ~/requirements_VCDemo.txt
