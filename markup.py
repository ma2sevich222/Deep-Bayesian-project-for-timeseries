

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from backtesting import Backtest
import backtesting._plotting as plt_backtesting
from utilits.make_df_from_csv_for_forward import make_df_from_csv_for_forward
import warnings
from tqdm import tqdm
import os

from utilits.markupSlide import markupSlide
from utilits.lazy_strategy import LazyStrategy

if not os.path.isdir('mark_up_results'):
    os.mkdir('mark_up_results')
if not os.path.isdir('mark_up_labeled_outputs'):
    os.mkdir('mark_up_labeled_outputs')

plt_backtesting._MAX_CANDLES = 1_000_000
pd.pandas.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)
pd.set_option("display.precision", 2)
warnings.filterwarnings("ignore")
filename = 'GC_2020_2022_15min.csv'
df = make_df_from_csv_for_forward(f'source_root/{filename}')


# Extracting Features for the Model from Continuous Data
"""lowpass filter (LF)"""
# https://medium.com/analytics-vidhya/how-to-filter-noise-with-a-low-pass-filter-python-885223e5e9b7
from scipy.signal import butter, filtfilt
def butter_lowpass_filter(data, cutoff_ratio, order=2):  # 57/100
    if cutoff_ratio/100 >= 1:
        print('\ncutoff_ratio >= 1, поэтому сглаживание невозможно!!!')
        return data
    b, a = butter(order, cutoff_ratio/100, btype='low', analog=False)
    y = filtfilt(b, a, data, axis=0)
    return y


# df = pd.read_pickle(source_filename).sort_index()
# df = pd.read_csv(source_filename, index_col='Datetime').sort_index()
# del df['Unnamed: 0']
# # df = df.reset_index().drop_duplicates(subset='Datetime', keep='last').set_index(
# #     'Datetime')  # удаление дубликатов по индексу
# df_duplicated = df[df.index.duplicated(keep=False)].sort_index()  # проверка дубликатов
# assert df_duplicated.shape[0] == 0, "В коде существуют дубликаты!"
# df.index = pd.to_datetime(df.index)
# print(f'Датасет: {df.shape[0]}\n\n')


""""""""" Подбор разметки для выдерживания комиссии """""""""
results = pd.DataFrame(columns=['extrW', 'FinalEquity', 'SharpRatio'], index=[])
finalEq = []
sharpR = []

NQ_list = [10,20,30,40,50,60,70,80,90,95,100]  # 5 - оставили 5% от составляющих сигнала (Убрали 95% шума) (Чем меньше, тем лучше сглаживание)

# показать Backtest на определенном extrW и NQ
showBacktest = True
nq = 90
extW = 4

if showBacktest:
    df[nq] = butter_lowpass_filter(df['Close'], cutoff_ratio=nq, order=2)
    data = markupSlide(df, smoothing=nq, extrW=extW, commission=4.62, checking=True)
    data = data.drop(nq, axis=1)
    bt = Backtest(data, strategy=LazyStrategy,
                  cash=100000, commission_type="absolute", commission=0,
                  features_coeff=10, exclusive_orders=True)
    stats = bt.run(deal_amount='fix')
    bt.plot()

for idx, NQ in enumerate(tqdm(NQ_list)):

    finalEq.append([])
    sharpR.append([])

    for extrW in range(1, 10):
        df[NQ] = butter_lowpass_filter(df['Close'], cutoff_ratio=NQ, order=2)
        data = markupSlide(df, smoothing=NQ, extrW=extrW, commission=4.62, checking=True)
        data = data.drop(NQ, axis=1)
        bt = Backtest(data, strategy=LazyStrategy,
                      cash=100000, commission_type="absolute", commission=4.62,
                      features_coeff=10, exclusive_orders=True)
        stats = bt.run(deal_amount='fix')
        data.to_csv(f'mark_up_labeled_outputs/{filename[:-4]}_nq{NQ}_extr{extrW}.csv')
        row = {'extrW': extrW, 'FinalEquity': stats[4], 'SharpRatio': stats[10]}
        row = pd.DataFrame(row, index=[0])
        finalEq[idx].append(stats[4])
        sharpR[idx].append(stats[10])

    results.to_csv(f'mark_up_results/results_NQ{NQ}.csv')


for i in range(len(NQ_list)):
    plt.plot(range(1, 1+len(finalEq[i])), finalEq[i], label=f'NQ{NQ_list[i]}')

plt.xlabel('extrW')
plt.ylabel('Final Equity')
plt.title("Подбор сглаживания и шага Final Equity")
plt.legend()
plt.grid()
plt.show()

for i in range(len(NQ_list)):
    plt.plot(range(1, 1 + len(sharpR[i])), sharpR[i], label=f'NQ{NQ_list[i]}')

plt.xlabel('extrW')
plt.ylabel('Sharp Ratio')
plt.title("Подбор сглаживания и шага Sharp Ratio")
plt.legend()
plt.grid()
plt.show()
