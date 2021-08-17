import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

frequency_sampling, audio_signal = wavfile.read\
    ("C:/Users/swagata/OneDrive - Hiroshima University/personal/Job/Python_AI_practice/Python_AI_prac/audio_file.wav")

# display parameters like sampling frequency, data type
print('\nSignal shape:', audio_signal.shape)
print('Signal Datatype:', audio_signal.dtype)
print('Signal duration', round(audio_signal.shape[0]/
                               float(frequency_sampling), 2), 'seconds')

# normalize the signal
audio_signal = audio_signal/np.power(2,15)

# extract first 100 values from this signal to visualize
audio_signal = audio_signal[:100]
time_axis = 1000*np.arange(0,len(audio_signal), 1)/ float(frequency_sampling)

# visualize the signal
plt.plot(time_axis, audio_signal, color='blue')
plt.xlabel('Time (milliseconds)')
plt.ylabel('Amplitude')
plt.title('Input audio signal')
plt.show()

# extract the length of audio signal
length_signal = len(audio_signal)
half_length = np.ceil((length_signal+1)/2.0).astype(np.int)

# convert signal to frequency domain
signal_frequency = np.fft.fft(audio_signal)

# normalize the freq domain signal and square it
signal_frequency = abs(signal_frequency[0:half_length]) / length_signal
signal_frequency **= 2

# extract the length and half length of the frequency transformed signal
len_fts = len(signal_frequency)

# adjust the FT signal for even and odd cases
if length_signal % 2:
    signal_frequency[1:len_fts] *= 2
else:
    signal_frequency[1:len_fts-1] *= 2

# extract the power in decibel
signal_power = 10 * np.log10(signal_frequency)

# adjust the frequency in kHz for X-axis
x_axis = np.arange(0, half_length, 1) * (frequency_sampling/ length_signal) / 1000.0

# visualize the data
plt.figure()
plt.plot(x_axis, signal_power, color='black')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Signal power (dB)')
plt.show()