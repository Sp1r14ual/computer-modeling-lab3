import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy.fft import fft, ifft
from sklearn.metrics import mean_squared_error
from scipy import stats
import os

# Создаем папку для графиков
os.makedirs('plots', exist_ok=True)

# 1. Загрузка данных


def load_data():
    df = pd.read_csv('data.csv')
    # Конвертируем в float на случай строкового формата
    y = df['y'].values.astype(float)
    return {'y': y}

# 2. Преобразования данных


def apply_transformations(series):
    diff1 = np.diff(series, n=1)
    diff2 = np.diff(diff1, n=1)
    return {
        'original': series,
        'first_diff': diff1,
        'second_diff': diff2
    }

# 3. Анализ Фурье


def fourier_analysis(data, L):
    fft_vals = fft(data[:L])
    amplitudes = np.abs(fft_vals) / L
    return amplitudes[:L//2]

# 4. Прогнозирование


def fourier_forecast(data, num_harmonics, forecast_steps):
    fft_vals = fft(data)
    sorted_indices = np.argsort(np.abs(fft_vals))[::-1]
    fft_vals_filtered = np.zeros_like(fft_vals)
    fft_vals_filtered[sorted_indices[:num_harmonics]
                      ] = fft_vals[sorted_indices[:num_harmonics]]
    forecast = np.real(ifft(fft_vals_filtered, n=len(
        data)+forecast_steps))[-forecast_steps:]
    return forecast

# 5. Генерация графиков


def generate_plots(series_name, data_dict, window_ratios):
    n = len(data_dict['original'])

    # Графики 1-3: Исходные данные и разности
    for transform in ['original', 'first_diff', 'second_diff']:
        plt.figure(figsize=(10, 4))
        plt.plot(data_dict[transform])
        plt.title(
            f'{series_name.upper()} - {transform.replace("_", " ").title()}')
        plt.savefig(f'plots/{series_name}_{transform}.png')
        plt.close()

    # Графики 4-15: Спектры Фурье
    for transform in ['original', 'first_diff', 'second_diff']:
        series = data_dict[transform]
        for ratio in window_ratios:
            L = int(len(series) * ratio)
            if L < 10:
                continue

            amplitudes = fourier_analysis(series, L)
            freq = np.fft.fftfreq(L)[:L//2]

            plt.figure(figsize=(10, 4))
            plt.plot(freq, amplitudes)
            plt.title(f'{series_name.upper()} {transform.replace(
                "_", " ").title()} - Spectrum (L={L})')
            plt.savefig(
                f'plots/{series_name}_{transform}_spectrum_{ratio}.png')
            plt.close()

    # Графики 16-27: Прогнозы
    for transform in ['original', 'first_diff', 'second_diff']:
        series = data_dict[transform]
        for ratio in window_ratios:
            L = int(len(series) * ratio)
            if L < 10 or L + 5 > len(series):
                continue

            # Выбор гармоник
            amplitudes = fourier_analysis(series, L)
            threshold = np.mean(amplitudes) + np.std(amplitudes)
            num_harmonics = np.sum(amplitudes > threshold)

            # Прогноз
            forecast = fourier_forecast(series[:L], num_harmonics, 5)
            actual = series[L:L+5]

            plt.figure(figsize=(10, 4))
            plt.plot(np.arange(L), series[:L], label='Train')
            plt.plot(np.arange(L, L+5), forecast, 'r--', label='Forecast')
            plt.plot(np.arange(L, L+5), actual, 'g', label='Actual')
            plt.title(f'{series_name.upper()} {transform.replace(
                "_", " ").title()} - Forecast (L={L})')
            plt.legend()
            plt.savefig(
                f'plots/{series_name}_{transform}_forecast_{ratio}.png')
            plt.close()


# Основной код
data = load_data()
window_ratios = [0.25, 0.5, 0.75, 0.9]

# Обрабатываем только временной ряд 'y'
for series_name in ['y']:
    original_series = data[series_name]
    transformed_data = apply_transformations(original_series)
    generate_plots(series_name, transformed_data, window_ratios)

print("Все графики сохранены в папку 'plots'")
