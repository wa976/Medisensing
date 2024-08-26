import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch

def calculate_snr(audio_signal):
    # 신호의 파워 계산
    signal_power = np.mean(audio_signal**2)
    
    # 노이즈의 파워 추정 (신호의 하위 10%를 노이즈로 가정)
    noise_power = np.mean(np.sort(audio_signal**2)[:len(audio_signal)//10])
    
    # SNR 계산
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def check_lung_sound_quality(file_path, snr_threshold=10):
    # WAV 파일 읽기
    sample_rate, audio_data = wavfile.read(file_path)
    
    # 스테레오인 경우 모노로 변환
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # SNR 계산
    snr = calculate_snr(audio_data)
    
    # 임계값과 비교하여 결과 반환
    return 0 if snr >= snr_threshold else 1, snr

def process_folder(folder_path, snr_threshold=10):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            quality, snr = check_lung_sound_quality(file_path, snr_threshold)
            results.append((filename, quality, snr))
    return results

# 사용 예시
folder_path = './data4/797)1-7-W_I.wav'
results = process_folder(folder_path)

for filename, quality, snr in results:
    print(f"파일: {filename}, 품질: {'양호' if quality == 0 else '불량'}, SNR: {snr:.2f} dB")