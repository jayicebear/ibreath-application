from google.colab import drive
drive.mount('/content/drive')

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import firestore
from firebase_admin import storage
if not firebase_admin._apps:
    cred = credentials.Certificate('/content/drive/MyDrive/Mozzign ML team/ibreath2live-firebase-adminsdk-nkhmj-1bfdabd032.json') 
    default_app = firebase_admin.initialize_app(cred, {'databaseURL':'https://ibreath2live.firebaseio.com/'})
   
db = firestore.client()

import os

audioFilePaths = []
recordingInfoFilePaths = []
for dirname, _, filenames in os.walk('/content/drive/MyDrive/Mozzign ML team/breathfolder'):
    for filename in filenames:
        fullPath = os.path.join(dirname, filename)
        if filename.endswith("wav"):
            audioFilePaths.append(fullPath)
        elif filename.endswith("txt"):
            recordingInfoFilePaths.append(fullPath) 
        #print(os.path.join(dirname, filename))
recordingInfoFilePaths = mode
print(len(audioFilePaths))
print(len(recordingInfoFilePaths))

import librosa
import numpy as np

# 이 부분은 숫자 임의대로 바꾸어줘도 상관없음
gSampleRate = 7000

def loadFiles(fileList):
    outputBuffers = []
    for filename in fileList:
    # audioButter: 소리데이터 에서 숫자로 변형된 백터
      # nativeSampleRate: hz (보통 44100hz가량)
        audioBuffer, nativeSampleRate = librosa.load(filename, dtype=np.float32, mono=True, sr=None)
        print(len(audioBuffer))  
        if nativeSampleRate == gSampleRate:
            outputBuffers.append(audioBuffer)
        else: #1초에 7000이 되도록 설정해줌
        #Linear resampling using numpy is significantly faster than Librosa's default technique
            # duration은 총 녹음의 길이라고 보면 됨 ex) 882000(한 파일당 전체 데이터 수)/44100(1초) = 20 초
            duration = len(audioBuffer) / nativeSampleRate
            #타겟으로 하는 샘플의 개수(데이터의 총 백터) -- 그래프를 어느정도로 자세하게 그릴거냐 하는 정도
            nTargetSamples = int(duration * gSampleRate)
            #대부분 20초 인데 20초를 백터의 길이만큼 나누어(?)줌 -- x 축
            timeXSource = np.linspace(0, duration, len(audioBuffer), dtype=np.float32)
            # 20초 를 7000HZ*20초 개로 나누어서 백터로  
            # 왜냐하면 1초에 데이터 1개가 아니라 1초를 7000개의 hz 로 나타내기 위하여
            #[0.0000000e+00 1.4285816e-04 2.8571632e-04 ... 1.9999714e+01 1.9999857e+01, 2.0000000e+01]
            timeX = np.linspace(0, duration, nTargetSamples, dtype=np.float32)
            # timeX: 보간된 값을 평가할 x 좌표, timeXSource : x축, audioBuffer: y축
            resampledBuffer = np.interp(timeX, timeXSource, audioBuffer)
            outputBuffers.append(resampledBuffer)
            print(resampledBuffer)
    return outputBuffers

#audioBuffers는 각 음성파일이 벡터로 변환된 형태
audioBuffers = loadFiles(audioFilePaths)


from scipy import signal
import matplotlib.pyplot as plt

upperCutoffFreq = 3000
cutoffFrequencies = [80, upperCutoffFreq]

#FIR coefficients for a bandpass filter with a window of 80-3000 Hz
#
highPassCoeffs = signal.firwin(241, cutoffFrequencies, fs=gSampleRate, pass_zero="bandpass")

def applyHighpass(npArr):
    return signal.lfilter(highPassCoeffs, [1.0], npArr)

#Scales all samples to ensure the peak signal is 1/-1
def normalizeVolume(npArr):
    #최대와 최소 amp peak찾기
    minAmp, maxAmp = (np.amin(npArr), np.amax(npArr))
   	#노멀라이즈를 위해서는 음수든 양수든 최대의 값을 찾아야 하므로 abs 후 최대값 구하기
    maxEnv = max(abs(minAmp), abs(maxAmp))
		#구한 최대값이 1이어야 하므로 ratio = 1/max_peak
    scale = 1.0 / maxEnv
    print(scale)
    #in place multiply/ 노멀라이즈 한 값으로 변환한 배열
    npArr *= scale
    return npArr

#Higher gamma results in more aggressive compression
def applyLogCompressor(signal, gamma):
    sign = np.sign(signal) #시그널은 진폭 'a'를 가진 사인곡선
    absSignal = 1 + np.abs(signal) * gamma #로그 변환 공식(Γr(v):=log(1+r⋅v)) 
    logged = np.log(absSignal) 
    scaled = logged * (1 / np.log(1.0 + gamma)) #Divide by the maximum possible value from compression
    #scaled는 compressed output, loggedsms compression factor)
    return sign * scaled #로그압축공식(sign(x)*scaled)
    
"""
scale: 4.3225139378512765  scale: 0.6756259522726079  scale: 1.430938662968741

npArr: [-7.57816278e-09  5.36031410e-09  1.79346926e-08 ...  9.72197264e-04
  3.18080176e-04 -2.72038422e-04]
npArr: [0.         0.         0.         ... 0.41142793 0.41966315 0.43201317]
npArr: [0.         0.         0.         ... 0.00890557 0.01237403 0.01870259]

npAPP*=sacle [-3.27567142e-08  2.31700324e-08  7.75229589e-08 ...  4.20233622e-03
  1.37490600e-03 -1.17588987e-03]
npAPP*=sacle [0.         0.         0.         ... 0.27797139 0.28353532 0.29187931]
npAPP*=sacle [0.         0.         0.         ... 0.01274332 0.01770647 0.02676226]
"""

#Removing the low-freq noise, re-normalizing volume then apply compressor
noiseRemoved = [normalizeVolume(applyHighpass(buffer)) for buffer in audioBuffers] #각 버퍼에서 highpass를 통해서 low-frequency 없애고 노멀라이즈
noiseRemoved = [applyLogCompressor(sig, 30) for sig in noiseRemoved]               #그 후 compressor적용


windowSizeSeconds = 0.05 
windowSampleSize = int(gSampleRate * windowSizeSeconds)

def plotSpectrogram(specData):
    plt.figure(figsize=(16,5))
    #Gamma scaling factor of 0.1 needed to make spectrogram more readable
    plt.pcolormesh(specData[1], specData[0], np.power(specData[2],0.1) , shading='gouraud')
    plt.ylim(0, upperCutoffFreq)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    
#(audioBuffers:float[][]) => (frequencies:float[], time(seconds):float[], amplitudes:float[][]))[]
def getSpectrograms(audioBuffers):
    spectrograms = []
    for buffer in audioBuffers:
        freqTable, times, powerSpectrum = signal.spectrogram(buffer, gSampleRate, nperseg=windowSampleSize)
        spectrograms.append((freqTable, times, powerSpectrum))
    return spectrograms

spectrograms = getSpectrograms(noiseRemoved)
plotSpectrogram(spectrograms[0])

#(spectrogram:float[][], cutoffFreq(hz):float, plot:bool) => (times:float, amplitudes:float[])
def getPowerEnvelop(spectrogram, cutoff, plot=False):
    frequencies = spectrogram[0]
    timeSlices = spectrogram[1]
    spectrum = spectrogram[2]
    
    maxInd = np.sum(frequencies <= cutoff)
    truncFreq = frequencies[:maxInd]
    
    powerEnvelop = []
    for idx, _ in enumerate(timeSlices):
        freqAmplitudes = spectrum[:maxInd,idx]
        
        powerBins = freqAmplitudes * np.square(truncFreq)
        powerEnvelop.append(sum(powerBins))
    if (plot): 
        plt.figure(figsize=(16,5))
        plt.title("Intensity vs time")
        plt.plot(timeSlices, powerEnvelop)
        plt.xlabel("Time(s)")
        plt.ylabel("Power")
        plt.show()
        
    return (timeSlices, powerEnvelop)

time, amp = getPowerEnvelop(spectrograms[0], upperCutoffFreq, True)


from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.signal import peak_widths
import math

#(amplitudes:float[], time(seconds):float[], sampleInterval(seconds):float, minPeakDuration(seconds):float, gaussainSmoothingSigma:float, peakRelHeight(0-1):float, plot:bool) =>
#(smoothed:float[], peakTiming(seconds):float[], leftRightBoundaries:(left(seconds):float, right(seconds):float)[])
def findPeaksAndWidthsFromSmoothedCurve(amplitudes,time, sampleInterval, minPeakDuration=0.4, gaussianSmoothingSigma = 3, peakRelHeight=0.8, plot=False):
    smoothed = gaussian_filter1d(amplitudes, gaussianSmoothingSigma)
    minPeakDurationSamples = int(math.ceil(minPeakDuration / sampleInterval))
    peakIndices, _ = find_peaks(smoothed, width=minPeakDurationSamples) 
    peakWidthResult = peak_widths(smoothed, peakIndices, peakRelHeight)
    
    leftPeakTimes = time[np.rint(peakWidthResult[2]).astype(int)]
    rightPeakTimes = time[np.rint(peakWidthResult[3]).astype(int)]
    leftRightBoundaries = list(zip(leftPeakTimes, rightPeakTimes))
    
    peakTiming = time[peakIndices]
    if plot:
        plt.figure(figsize=(16,5))
        plt.plot(time, amplitudes, color="tab:gray", label="Original Signal") 
        plt.plot(time, smoothed, color="tab:orange", label="Smoothed")
        plt.plot(peakTiming, smoothed[peakIndices], "v", color="red", markersize=10)
        plt.hlines(peakWidthResult[1], leftPeakTimes , rightPeakTimes , color="red")
        plt.xlabel("Time(s)")
        plt.ylabel("Intensity")
        plt.title("Peak Locations and Width (Red Markers)")
        plt.legend()
        plt.show()
        
    return (smoothed, peakTiming, leftRightBoundaries)
    

_  = findPeaksAndWidthsFromSmoothedCurve(amp, time, windowSizeSeconds, plot=True)
