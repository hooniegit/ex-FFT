import torch
import torch.nn as nn
import numpy as np  # 삼각함수 데이터 만드는 용도
import matplotlib.pyplot as plt # 그래프 그리는 용도 
import torch.fft as fft

# 데이터셋 생성
# 1000개의 데이터, 5번 진동 => 주기 = 250
def generate_data(seq_length=1000, n=0):
    time_steps = np.linspace(0, np.pi*8, seq_length)
    noise = np.random.randn(1000)*0.2 if n==1 else 0
    data = np.sin(time_steps) + noise
    data = data.reshape(-1, 1)
    return torch.tensor(data, dtype=torch.float32)

dataset_noise = generate_data(n=1)
dataset_clear = generate_data()
dataset_clear.reshape(-1)

import matplotlib.pyplot as plt
from decimal import Decimal
import tensorflow as tf

import matplotlib.pyplot as plt
import tensorflow as tf

# DFT(Discrete Fourier Transform) 수행
# DFT : time domain 함수를 frequency domain 함수로 변형
dft = fft.rfft(dataset_noise, dim=0)

# n = len(dataset)
n = dataset_noise.shape[0]

# 주파수 범위 구하기
freq_range = fft.rfftfreq(n, d=1)

# Fourier 변환 데이터의 절댓값 중 최댓값
max_idx = torch.argmax(torch.abs(dft))
print(max_idx)

# 최댓값을 가지는 곳의 주파수
max_freq = freq_range[max_idx]
period = 1/max_freq
period_int = int(tf.cast(tf.round(period), dtype=tf.int32))
print(period_int)

# 주파수 영역 생성
freqs = fft.rfftfreq(len(dataset_noise))

# 그래프 그리기
fig, ax = plt.subplots()
ax.plot(freqs[:len(freqs)//2], torch.abs(dft)[:len(freqs)//2])
ax.set_xlabel('Frequency')
ax.set_ylabel('Magnitude')
plt.show()
