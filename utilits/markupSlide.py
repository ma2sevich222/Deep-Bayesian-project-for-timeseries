
import numpy as np

def markupSlide(df, smoothing, extrW, commission=0, checking=False):
    """
    df - dataframe
    smoothing - Тип сглаживания
    extrW - Следующая n свеча для сравнения
    commission - комиссия со сделки

    Returns: - массив сигналов +1 покупай, -1 продавай
    """

    df = df.copy()
    prices = df[smoothing].values  # если нет сглаживания в разметке, то указать Close
    N = len(prices) - extrW

    out = np.zeros(N)
    for i in range(N):  # цикл по всем точкам
        current_price = prices[i]

        next_price = prices[i+extrW]

        if next_price - current_price - commission > 0:
            out[i] = 1
        elif current_price - next_price - commission > 0:
            out[i] = -1
        else:
            out[i] = 0


    out = np.append(out, [np.nan]*extrW)

    df['Signal'] = out

    # Меняем разметку к [-1, 1] - к двум классам
    df['Signal'].replace(0, np.nan, inplace=True)
    df['Signal'] = df['Signal'].ffill()
    df['Signal'] = df['Signal'].bfill()
    if df['Signal'].isnull().values.any():
        df['Signal'] = 1.



    # df.dropna(axis=0, inplace=True)  # Удаляем наниты
    # df = df.loc[df['Signal'] != 0]  # оставим только не нулевые строки

    # if state:
    #     df['State'] = df['Signal'].diff()
    #     df.loc[df.index[0], 'State'] = df.loc[df.index[0], 'Signal']
    #     df['State'].replace(2, 1, inplace=True)
    #     df['State'].replace(-2, -1, inplace=True)
        # print(pd.unique(df['State']))


    # # проверим прибыльность разметки
    # if checking:
    #     # Вариант с торговлей на фиксированную сумму
    #     bt = Backtest(df[:1000], LnS, cash=4000, commission=commission, trade_on_close=True)
    #     stats = bt.run(deal_amount='fix', fix_sum=2000)
    #     bt.plot(plot_volume=True, relative_equity=True)
    #     # print(stats)

    return df
