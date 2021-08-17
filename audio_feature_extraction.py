import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank

# read stored audio file
frequency_sampling, audio_signal = wavfile.read\
    ("C:/Users/swagata/OneDrive - Hiroshima University/personal/Job/Python_AI_practice/Python_AI_prac/audio_file.wav")

# take first 15000 samples for analysis
audio_signal = audio_signal[:15000]

# use MFCC techniques and extract MFCC features
features_mfcc = mfcc(audio_signal, frequency_sampling)

# print MFCC parameters
print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
print('Length of each feature =', features_mfcc.shape[1])

# plot and visualize MFCC features
features_mfcc = features_mfcc.T
plt.matshow(features_mfcc)
plt.title('MFCC')

# extract the filter bank features
filterbank_features = logfbank(audio_signal, frequency_sampling)

# print filterbank parameters
print('\nFilter bank:\nNumber of windows =', filterbank_features.shape[0])
print('Length of each feature =', filterbank_features.shape[1])

# plot and visualize the filterbank features.
filterbank_features = filterbank_features.T
plt.matshow(filterbank_features)
plt.title('Filter bank')
plt.show()