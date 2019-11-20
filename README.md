# VoiceConversionDemo

First download the trained models and save them somewhere.

Next check voice_conversion.py and change 'model_path' to where you save those models

How to use it: python voice_conversion.py --wav_path inputsample/mysampleinput1.m4a --model obama_neural --parallel

It will take 5-10 seconds to load the model and another few seconds to translate the input. The output will be stored as "result2.wav" in this directory.
