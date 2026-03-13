import numpy as np
import matplotlib.pyplot as plt
import scipy

def spectrogram_shorttimefft(wav_path, window_size, hop=10):
    fs, audio = scipy.io.wavfile.read(wav_path)
    window = scipy.signal.windows.hann(window_size)
    stft = scipy.signal.ShortTimeFFT(window, hop=hop, fs=fs)
    stft = np.abs(stft.stft(audio))

    return stft, fs, len(audio)

def plot_spectrogram(stft, fs, audio_len):
    plt.figure()
    plt.imshow(stft, aspect='auto', origin='lower', extent=[0, audio_len/fs, 0, fs/2])
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram (ShortTimeFFT)')
    plt.ylim(0, 1000)
    plt.grid(axis='y', alpha=0.5)
    plt.yticks(np.arange(0, 1001, 100))
    plt.show()


# Example usage
wav_path = 'progression.wav'  # replace with your WAV file
stft0, fs, audio_len = spectrogram_shorttimefft(wav_path, 500)
stft1, _, _ = spectrogram_shorttimefft(wav_path, 1000)
stft2, _, _= spectrogram_shorttimefft(wav_path, 2000)

plot_spectrogram(stft0, fs, audio_len)
plot_spectrogram(stft1, fs, audio_len)
plot_spectrogram(stft2, fs, audio_len)