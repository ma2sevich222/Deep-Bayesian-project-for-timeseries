
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import FunctionTransformer


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))
def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

def _heiken_ashi_transformer(data: pd.DataFrame) -> pd.DataFrame:
    """Трансформер для перевода обычных свечь в Heiken Ashi"""
    data['HA_Open'] = ((data['Open'] + data['Close']) / 2).shift(1)
    data['HA_Close'] = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
    data['HA_High'] = data[['High', 'HA_Open', 'HA_Close']].max(axis=1)
    data['HA_Low'] = data[['Low', 'HA_Open', 'HA_Close']].min(axis=1)

    data = data[list(data.columns)[:-4] + ['HA_Open', 'HA_High', 'HA_Low', 'HA_Close']]

    return data


def make_df_from_csv_for_forward(path):
    df = pd.read_csv(path, index_col='Datetime').sort_index()
    # del df['Unnamed: 0']
    # df = df.reset_index().drop_duplicates(subset='Datetime', keep='last').set_index(
    #     'Datetime')  # удаление дубликатов по индексу
    df_duplicated = df[df.index.duplicated(keep=False)].sort_index()  # проверка дубликатов
    assert df_duplicated.shape[0] == 0, "В коде существуют дубликаты!"

    df.index = pd.to_datetime(df.index)
    df['Date'] = df.index
    df['Date'] = df['Date'].dt.floor("D")
    numbarsperday = df.query("'2022-01-31' == Date")
    print(f'Число баров в торговом дне: {numbarsperday.shape[0]}')

    date = pd.to_datetime(df.Date)
    # df['year'] = date.apply(lambda x: x.year)
    # df['month'] = date.apply(lambda x: x.month)
    # df['week'] = date.apply(lambda x: x.week)
    # df['week_day'] = date.apply(lambda x: x.weekday())
    # df['day'] = date.apply(lambda x: x.day)
    # df['hour'] = date.apply(lambda x: x.hour)
    # df['minute'] = date.apply(lambda x: x.minute)
    # то: что дальше с датой - работает плохо
    # df["day_sin"] = sin_transformer(30).fit_transform(df["day"])
    # df["day_cos"] = cos_transformer(30).fit_transform(df["day"])
    # df["week_day_sin"] = sin_transformer(7).fit_transform(df["week_day"])
    # df["week_day_cos"] = cos_transformer(7).fit_transform(df["week_day"])
    # df["hour_sin"] = sin_transformer(24).fit_transform(df["hour"])
    # df["hour_cos"] = cos_transformer(24).fit_transform(df["hour"])
    # df["minute_sin"] = sin_transformer(60).fit_transform(df["minute"])
    # df["minute_cos"] = cos_transformer(60).fit_transform(df["minute"])
    del df['Date']  # df["day"], df['week_day'], df['hour'], df['minute']

    # df['meanOHCL'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    # df = _heiken_ashi_transformer(df)
    # df['volatility5'] = df['Close'].rolling(5).std()
    # df['volatility10'] = df['Close'].rolling(10).std()
    # df['volatility15'] = df['Close'].rolling(15).std()
    # df['volatility20'] = df['Close'].rolling(20).std()
    # df['volatility30'] = df['Close'].rolling(30).std()
    # df['volatility60'] = df['Close'].rolling(60).std()
    # df['volatility120'] = df['Close'].rolling(120).std()
    # df['volatility180'] = df['Close'].rolling(180).std()
    # df['roll4'] = df['volatility5'].rolling(4).corr(df['volatility30'])

    """SMA"""
    # df['Close_SMA2'] = df['Close'].rolling(2).mean()
    # df['Close_SMA3'] = df['Close'].rolling(3).mean()
    # df['Close_SMA4'] = df['Close'].rolling(4).mean()
    # df['Close_SMA5'] = df['Close'].rolling(5).mean()
    # df['Close_SMA10'] = df['Close'].rolling(10).mean()
    # df['Close_SMA15'] = df['Close'].rolling(15).mean()
    # df['Close_SMA30'] = df['Close'].rolling(30).mean()
    # df['Close_SMA60'] = df['Close'].rolling(60).mean()
    """SWA"""
    # df['Close_SWA2'] = df['Close'].ewm(span=2).mean()
    # df['Close_SWA3'] = df['Close'].ewm(span=3).mean()
    # df['Close_SWA4'] = df['Close'].ewm(span=4).mean()
    # df['Close_SWA5'] = df['Close'].ewm(span=5).mean()
    # df['Close_SWA10'] = df['Close'].ewm(span=10).mean()
    # df['Close_SWA15'] = df['Close'].ewm(span=15).mean()
    # df['Close_SWA30'] = df['Close'].ewm(span=30).mean()
    # df['Close_SWA60'] = df['Close'].ewm(span=60).mean()
    # """momentum Close"""
    # df['CloseShift1'] = np.log(df['Close'] / df['Close'].shift(1))
    # df['CloseShift2'] = np.log(df['Close'] / df['Close'].shift(2))
    # df['CloseShift3'] = np.log(df['Close'] / df['Close'].shift(3))
    # df['CloseShift4'] = np.log(df['Close'] / df['Close'].shift(4))
    # df['CloseShift10'] = np.log(df['Close'] / df['Close'].shift(10))
    # df['CloseShift15'] = np.log(df['Close'] / df['Close'].shift(15))
    # df['CloseShift30'] = np.log(df['Close'] / df['Close'].shift(30))
    # df['CloseShift60'] = np.log(df['Close'] / df['Close'].shift(60))
    """momentum Volume"""
    # df['VolumeShift1'] = np.log(df['Volume'] / df['Volume'].shift(1))
    # df['VolumeShift2'] = np.log(df['Volume'] / df['Volume'].shift(2))
    # df['VolumeShift3'] = np.log(df['Volume'] / df['Volume'].shift(3))
    # df['VolumeShift4'] = np.log(df['Volume'] / df['Volume'].shift(4))
    # df['VolumeShift10'] = np.log(df['Volume'] / df['Volume'].shift(10))
    # df['VolumeShift15'] = np.log(df['Volume'] / df['Volume'].shift(15))
    # df['VolumeShift30'] = np.log(df['Volume'] / df['Volume'].shift(30))
    # df['VolumeShift60'] = np.log(df['Volume'] / df['Volume'].shift(60))
    df.dropna(axis=0, inplace=True)  # Удаляем наниты

    return df