#######################################################
# Copyright © 2021-2099 Ekosphere. All rights reserved
# Author: Evgeny Matusevich
# Contacts: <ma2sevich222@gmail.com>
# File: forward.py
#######################################################
import pandas as pd
import plotly.express as px

file_root = "outputs/only_deep_b_GC_2020_2022_15min_nq90_extr4_11_08_2022"
filename = "intermedia_GC_2020_2022_15min_nq90_extr4.xlsx"
final_df = pd.read_excel(f"{file_root}/{filename}")  # загружаем результаты  анализа

df_plot = final_df[
    [
        "values_0",
        "values_1",
        "# Trades",
        "patch",
        "n_hiden",
        "n_hiden_two",
        "train_window",
        "forward_window",
    ]
]

fig = px.parallel_coordinates(
    df_plot,
    color="values_0",
    labels={
        "values_0": "Net Profit ($)",
        "values_1": "Sharpe Ratio",
        "# Trades": "Trades",
        "patch": "patch(bars)",
        "n_hiden": "n_neirons_1layer",
        "n_hiden_two": "n_neirons_2layer",
        "train_window": "train_window (bars)",
        "forward_window": "forward_window (bars)",
    },
    range_color=[df_plot["values_0"].min(), df_plot["values_0"].max()],
    color_continuous_scale=px.colors.sequential.Viridis,
    title=f"only_dbb_GC_2020_2022_15min_nq90_extr4",
)

fig.write_html(f"only_dbb_GC_2020_2022_15min_nq90_extr4.html")  # сохраняем в файл
fig.show()


'''
file_root = "outputs/net_deep_b_des_tree_GC_2020_2022_15min_nq90_extr4_11_08_2022"
filename = "intermedia_GC_2020_2022_15min_nq90_extr4.xlsx"
final_df = pd.read_excel(f"{file_root}/{filename}")  # загружаем результаты  анализа

df_plot = final_df[
    [
        "values_0",
        "values_1",
        "# Trades",
        "patch",

        "n_hiden",
        "n_hiden_two",
        "n_hiden_three",
        "n_hiden_four",
        "train_window",
        "forward_window",
    ]
]

fig = px.parallel_coordinates(
    df_plot,
    color="values_0",
    labels={
        "values_0": "Net Profit ($)",
        "values_1": "Sharpe Ratio",
        "# Trades": "Trades",
        "patch": "patch(bars)",

        "n_hiden": "n_neirons_1layer",
        "n_hiden_two": "n_neirons_2layer",
        "n_hiden_three":"n_neirons_3layer",
        "n_hiden_four":"n_neirons_4layer",
        "train_window": "train_window (bars)",
        "forward_window": "forward_window (bars)",
    },
    range_color=[df_plot["values_0"].min(), df_plot["values_0"].max()],
    color_continuous_scale=px.colors.sequential.Viridis,
    title=f"усложненная сеть_GC_2020_2022_15min_nq90_extr4",
)

fig.write_html(f"adv_net_intermedia_GC_2020_2022_15min_nq90_extr4.html")  # сохраняем в файл
fig.show()'''
