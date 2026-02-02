import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

bits_per_symbol = 4
M = 2 ** bits_per_symbol
num_symbols = 1000
sps = 8
fs = 1000
fc = 100

data = np.random.randint(0, 2, num_symbols * bits_per_symbol)
reshaped_data = data.reshape(num_symbols, bits_per_symbol)

symbol_indices = []
for i in range(num_symbols):
    bits = reshaped_data[i]
    index = 0
    for j in range(bits_per_symbol):
        index += bits[j] * (2 ** (bits_per_symbol - 1 - j))
    symbol_indices.append(index)

qam_constellation = np.zeros(M, dtype=complex)
side_len = int(np.sqrt(M))
for i in range(side_len):
    for j in range(side_len):
        qam_constellation[i * side_len + j] = (2 * i - side_len + 1) + 1j * (2 * j - side_len + 1)

qam_symbols = qam_constellation[symbol_indices]

t_symbol = np.arange(num_symbols)
plt.figure(figsize=(10, 6))
plt.scatter(np.real(qam_symbols), np.imag(qam_symbols), alpha=0.6)
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.title('نمودار صورت فلکی 16-QAM')
plt.xlabel('جزء حقیقی')
plt.ylabel('جزء موهومی')
plt.show()

upsampled = np.zeros(num_symbols * sps, dtype=complex)
upsampled[::sps] = qam_symbols

filter_taps = signal.firwin(101, 0.4)
filtered = signal.convolve(upsampled, filter_taps, mode='same')

t_mod = np.arange(len(filtered)) / fs
carrier = np.exp(1j * 2 * np.pi * fc * t_mod)
modulated = filtered * carrier

channel_noise = 0.1 * (np.random.randn(len(modulated)) + 1j * np.random.randn(len(modulated)))
modulated_noisy = modulated + channel_noise

received = modulated_noisy * np.conj(carrier)
received_filtered = signal.convolve(received, filter_taps, mode='same')

downsampled = received_filtered[::sps]

received_symbols = np.zeros(num_symbols, dtype=complex)
for i in range(num_symbols):
    distances = np.abs(qam_constellation - downsampled[i])
    received_symbols[i] = qam_constellation[np.argmin(distances)]

received_indices = [np.where(qam_constellation == sym)[0][0] for sym in received_symbols]

received_bits = []
for idx in received_indices:
    bits = [(idx >> (bits_per_symbol - 1 - j)) & 1 for j in range(bits_per_symbol)]
    received_bits.extend(bits)

received_bits = np.array(received_bits[:len(data)])
ber = np.sum(data != received_bits) / len(data)
print(f"نرخ خطای بیتی (BER): {ber:.6f}")

plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
plt.plot(np.real(filtered[:200]), label='Real')
plt.plot(np.imag(filtered[:200]), label='Imag')
plt.title('سیگنال پایه فیلتر شده')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(t_mod[:200], np.real(modulated[:200]))
plt.title('سیگنال مدوله شده (بخش حقیقی)')

plt.subplot(2, 3, 3)
plt.scatter(np.real(downsampled), np.imag(downsampled), alpha=0.6, c='red')
plt.grid(True)
plt.title('صورت فلکی دریافتی')

plt.subplot(2, 3, 4)
plt.plot(np.real(modulated_noisy[:200]), alpha=0.7)
plt.title('سیگنال دریافتی نویزی (حقیقی)')

plt.subplot(2, 3, 5)
plt.plot(np.real(received[:200]), label='Real')
plt.plot(np.imag(received[:200]), label='Imag')
plt.title('سیگنال پایه‌گذاری شده')
plt.legend()

plt.subplot(2, 3, 6)
plt.scatter(np.real(qam_symbols), np.imag(qam_symbols), alpha=0.3, label='ارسال شده')
plt.scatter(np.real(received_symbols), np.imag(received_symbols), alpha=0.3, label='دریافت شده', marker='x')
plt.grid(True)
plt.title('مقایسه صورت فلکی')
plt.legend()

plt.tight_layout()
plt.show()