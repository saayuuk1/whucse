import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import sklearn

def power_spectrogram():
    #读取音频文件
    y, sr = librosa.load('./export.wav')
    #准备绘图
    plt.figure()

    #做短时傅里叶变换，并提取频率的振幅，将其转换为dB标度频谱
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    #绘制线性频谱图
    librosa.display.specshow(D, y_axis='linear', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')
    plt.show()

    #绘制对数频谱图
    librosa.display.specshow(D, y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram')
    plt.show()

def mel_spectrogram():
    #读取音频文件
    y, sr = librosa.load('./export.wav')
    #准备绘图
    plt.figure(figsize=(10, 4))

    #计算得到Mel频谱
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    #绘制Mel频谱
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()

    #提取MFCC系数
    mfccs = librosa.feature.mfcc(y, sr)
    #绘制Mel倒谱
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.show()

    #MFCC特征缩放
    mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
    #绘制缩放后的Mel倒谱
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.show()

if __name__ == '__main__':
    power_spectrogram()
    mel_spectrogram()