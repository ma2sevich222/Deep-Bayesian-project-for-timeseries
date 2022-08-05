#######################################################
# Copyright © 2021-2099 Ekosphere. All rights reserved
# Author: Evgeny Matusevich
# Contacts: <ma2sevich222@gmail.com>
# File: functions.py
#######################################################
import pandas as pd
import backtesting._plotting as plt_backtesting
from backtesting import Backtest
from utilits.lazy_strategy import LazyStrategy
import numpy as np



def data_to_binary(train_df, forward_df, look_back):
    binary_train = train_df.iloc[:, 1:].diff()
    binary_train[binary_train < 0] = 0
    binary_train[binary_train > 0] = 1
    target = binary_train["Close"].values[2:]
    binary_train = binary_train[1:-1]
    binary_train["Target"] = target

    train_samples = [
        binary_train[i - look_back : i]
        for i in range(len(binary_train))
        if i - look_back >= 0
    ]
    Train_X = []
    Train_labels = []
    for sample in train_samples:
        Train_X.append(
            sample[["Open", "High", "Low", "Close", "Volume"]].to_numpy().flatten()
        )
        Train_labels.append(sample["Target"].iloc[-1])

    Train_Y = [[1, 0] if i == 0 else [0, 1] for i in Train_labels]

    binary_forward = forward_df.iloc[:, 1:].diff()
    binary_forward[binary_forward < 0] = 0
    binary_forward[binary_forward > 0] = 1
    binary_forward = binary_forward[1:]
    forward_df = forward_df[1:]
    foraward_samples = [
        forward_df[i - look_back : i]
        for i in range(len(forward_df))
        if i - look_back >= 0
    ]
    forward_binary_samples = [
        binary_forward[i - look_back : i]
        for i in range(len(binary_forward))
        if i - look_back >= 0
    ]
    Test_X = []
    Date = []
    Open = []
    High = []
    Low = []
    Close = []
    Volume = []
    for sample in forward_binary_samples:
        Test_X.append(
            sample[["Open", "High", "Low", "Close", "Volume"]].to_numpy().flatten()
        )
    for or_sample in foraward_samples:
        Date.append(or_sample["Datetime"].iloc[-1])
        Open.append(or_sample["Open"].iloc[-1])
        High.append(or_sample["High"].iloc[-1])
        Low.append(or_sample["Low"].iloc[-1])
        Close.append(or_sample["Close"].iloc[-1])
        Volume.append(or_sample["Volume"].iloc[-1])
    Signals = pd.DataFrame(
        {
            "Datetime": Date,
            "Open": Open,
            "High": High,
            "Low": Low,
            "Close": Close,
            "Volume": Volume,
        }
    )

    return np.array(Train_X), np.array(Train_Y), np.array(Test_X), Signals
def bayes_tune_get_stat_after_forward(
    result_df,
    lookback_size,
    epochs,
    n_hiden,
    n_hiden_two,
    train_window,
    forward_window,
    source_file_name,
    out_root,
    out_data_root,
    trial_namber,
    get_trade_info=False,
):
    plt_backtesting._MAX_CANDLES = 200_000
    pd.pandas.set_option("display.max_columns", None)
    pd.set_option("expand_frame_repr", False)
    pd.options.display.expand_frame_repr = False
    pd.set_option("display.precision", 2)

    result_df.set_index("Datetime", inplace=True)
    result_df.index = pd.to_datetime(result_df.index)
    result_df = result_df.sort_index()



    """ Параметры тестирования """
    i = 0
    deposit = 200000  # сумма одного контракта GC & CL
    comm = 4.6  # GC - комиссия для золота


    """ Тестирвоание """




    df_stats = pd.DataFrame()


    bt = Backtest(
        result_df,
        strategy=LazyStrategy,
        cash=deposit,
        commission_type="absolute",
        commission=4.62,
        features_coeff=10,
        exclusive_orders=True,
    )
    stats = bt.run()[:27]



    df_stats = df_stats.append(stats, ignore_index=True)
    df_stats["Net Profit [$]"] = (
        df_stats.loc[i, "Equity Final [$]"]
        - deposit
        - df_stats.loc[i, "# Trades"] * comm
    )
    # df_stats.loc[i, "buy_before"] = buy_before * step
    # df_stats.loc[i, "sell_after"] = sell_after * step
    df_stats["train_window"] = train_window
    df_stats["forward_window"] = forward_window
    df_stats["lookback_size"] = lookback_size
    df_stats["epochs"] = epochs
    df_stats["n_hiden"] = n_hiden
    df_stats["n_hiden_two"] = n_hiden_two

    if get_trade_info == True and df_stats["Net Profit [$]"].values > 0:
        bt.plot(
            plot_volume=True,
            relative_equity=False,
            filename=f"{out_root}/{out_data_root}/{trial_namber}_bt_plot_{source_file_name[:-4]}train_window{train_window}forward_window{forward_window}_lookback_size{lookback_size}.html",
        )
        stats.to_csv(
            f"{out_root}/{out_data_root}/{trial_namber}_stats_{source_file_name[:-4]}_train_window{train_window}forward_window{forward_window}_lookback_size{lookback_size}.txt"
        )
        result_df["Signal"] = result_df["Signal"].astype(int)

        result_df.insert(0, "Datetime", result_df.index)
        result_df = result_df.reset_index(drop=True)
        result_df[
            ["Datetime", "Open", "High", "Low", "Close", "Volume", "Signal"]
        ].to_csv(
            f"{out_root}/{out_data_root}/{trial_namber}_signals_{source_file_name[:-4]}_train_window{train_window}forward_window{forward_window}_lookback_size{lookback_size}.csv"
        )

    return df_stats



def labeled_data_to_binary(train_df, forward_df, look_back):
    binary_train = train_df[["Open", "High", "Low", "Close", "Volume"]].diff()
    binary_train=binary_train[1:]
    binary_train[binary_train < 0] = 0
    binary_train[binary_train > 0] = 1
    #target = binary_train["Close"].values[2:]
    #binary_train = binary_train[1:-1]
    #binary_train["Target"] = target
    binary_train["Target"] = train_df['Signal'].values[1:]
    binary_train.loc[binary_train["Target"] == -1, "Target"] = 0


    train_samples = [
        binary_train[i - look_back : i]
        for i in range(len(binary_train))
        if i - look_back >= 0
    ]
    Train_X = []
    Train_labels = []
    for sample in train_samples:
        Train_X.append(
            sample[["Open", "High", "Low", "Close", "Volume"]].to_numpy().flatten()
        )
        Train_labels.append(sample["Target"].iloc[-1])

    Train_Y = [[1, 0] if i == 0 else [0, 1] for i in Train_labels]

    binary_forward = forward_df[["Open", "High", "Low", "Close", "Volume"]].diff()
    binary_forward[binary_forward < 0] = 0
    binary_forward[binary_forward > 0] = 1
    binary_forward = binary_forward[1:]

    forward_df = forward_df[1:]
    foraward_samples = [
        forward_df[i - look_back : i]
        for i in range(len(forward_df))
        if i - look_back >= 0
    ]
    forward_binary_samples = [
        binary_forward[i - look_back : i]
        for i in range(len(binary_forward))
        if i - look_back >= 0
    ]
    Test_X = []
    Date = []
    Open = []
    High = []
    Low = []
    Close = []
    Volume = []
    for sample in forward_binary_samples:
        Test_X.append(
            sample[["Open", "High", "Low", "Close", "Volume"]].to_numpy().flatten()
        )
    for or_sample in foraward_samples:
        Date.append(or_sample["Datetime"].iloc[-1])
        Open.append(or_sample["Open"].iloc[-1])
        High.append(or_sample["High"].iloc[-1])
        Low.append(or_sample["Low"].iloc[-1])
        Close.append(or_sample["Close"].iloc[-1])
        Volume.append(or_sample["Volume"].iloc[-1])
    Signals = pd.DataFrame(
        {
            "Datetime": Date,
            "Open": Open,
            "High": High,
            "Low": Low,
            "Close": Close,
            "Volume": Volume,
        }
    )

    return np.array(Train_X), np.array(Train_Y), np.array(Test_X), Signals