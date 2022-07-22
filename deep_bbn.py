#######################################################
# Copyright © 2021-2099 Ekosphere. All rights reserved
# Author: Evgeny Matusevich
# Contacts: <ma2sevich222@gmail.com>
# File: deep_bbn.py
#######################################################
import pandas as pd

import optuna
import plotly.express as px

import os

import torch.optim as optim
import torchbnn as bnn

from datetime import date

from torch.utils.data import DataLoader
import numpy as np

import torch
import torch.nn as nn

from torchbnn.utils import freeze
from utilits.functions import bayes_tune_get_stat_after_forward,data_to_binary
from utilits.classes_and_models import DBNataset
from sklearn.tree import DecisionTreeClassifier

today = date.today()
source = "source_root"
out_root = "outputs"
source_file_name = "GC_2020_2022_60min.csv"
start_forward_time = "2021-01-04 00:00:00"
date_xprmnt = today.strftime("%d_%m_%Y")
out_data_root = f"deep_b_des_tree_{source_file_name[:-4]}_{date_xprmnt}"
os.mkdir(f"{out_root}/{out_data_root}")
intermedia = pd.DataFrame()
intermedia.to_excel(
    f"{out_root}/{out_data_root}/intermedia_{source_file_name[:-4]}.xlsx"
)
clf = DecisionTreeClassifier()
n_trials = 2


###################################################################################################




def objective(trial):
    start_forward_time = "2021-01-04 00:00:00"
    df = pd.read_csv(f"{source}/{source_file_name}")
    forward_index = df[df["Datetime"] == start_forward_time].index[0]


    """""" """""" """""" """""" """"" Параметры сети """ """""" """""" """""" """"""

    batch_s = 300


    """""" """""" """""" """""" """"" Параметры для оптимизации   """ """ """ """ """ """ """ """ """ ""

    patch = trial.suggest_int("patch", 2, 100)
    epochs = trial.suggest_int("epochs", 500, 1000)
    n_hiden = trial.suggest_int("n_hiden", 10, 300, step=10)
    n_hiden_two = trial.suggest_int("n_hiden_two", 10, 300, step=10)
    train_window = trial.suggest_categorical("train_window", [1000, 3000, 5000])
    forward_window = trial.suggest_categorical(
        "forward_window", [705, 1411, 2822, 5644]
    )
    ##############################################################################################
    DBNmodel = nn.Sequential(
        bnn.BayesLinear(
            prior_mu=0,
            prior_sigma=0.1,
            in_features=patch * 5,
            out_features=n_hiden,
        ),
        nn.ReLU(),
        bnn.BayesLinear(
            prior_mu=0, prior_sigma=0.1, in_features=n_hiden, out_features=n_hiden_two
        ),
        nn.ReLU(),
        bnn.BayesLinear(
            prior_mu=0, prior_sigma=0.1, in_features=n_hiden_two, out_features=2
        ),
    )
    ###################################################################################################
    df_for_split = df[(forward_index - train_window) :]
    df_for_split = df_for_split.reset_index(drop=True)
    n_iters = (len(df_for_split) - int(train_window)) // int(forward_window)

    signals = []
    for n in range(n_iters):

        train_df = df_for_split[:train_window]

        if n == n_iters - 1:
            forward_df = df_for_split[train_window:]
        else:
            forward_df = df_for_split[
                int(train_window) : sum([int(train_window), int(forward_window)])
            ]
        df_for_split = df_for_split[int(forward_window) :]
        df_for_split = df_for_split.reset_index(drop=True)
        Train_X, Train_Y, Forward_X, Signals = data_to_binary(
            train_df, forward_df, patch
        )

        DNB_dataset = DBNataset(
            Train_X[: len(Train_X) // 2], Train_Y[: len(Train_X) // 2]
        )
        DNB_dataloader = DataLoader(DNB_dataset, batch_size=batch_s, shuffle=False)
        cross_entropy_loss = nn.CrossEntropyLoss()
        klloss = bnn.BKLLoss(reduction="mean", last_layer_only=False)
        klweight = 0.01
        optimizer = optim.Adam(DBNmodel.parameters(), lr=0.001)

        for step in range(epochs):
            for _, (data, target) in enumerate(DNB_dataloader):

                models = DBNmodel(data)
                cross_entropy = cross_entropy_loss(models, target)

                kl = klloss(DBNmodel)
                total_cost = cross_entropy + klweight * kl



                optimizer.zero_grad()
                total_cost.backward()
                optimizer.step()

            if step % 100 == 0:
                    print("Энтропия", cross_entropy)
                    print("Финальная лосс", total_cost)

        DT_trainX = []
        DBNmodel.eval()
        freeze(DBNmodel)
        with torch.no_grad():
            for arr in Train_X[len(Train_X) // 2 :]:
                arr = torch.from_numpy(arr.astype(np.float32))

                pred = DBNmodel(arr)
                DT_trainX.append(
                    [
                        float(torch.argmax(pred).cpu().detach().numpy()),
                        float(pred[torch.argmax(pred)].cpu().detach().numpy()),
                    ]
                )

        des_lable = [np.argmax(i) for i in Train_Y[len(Train_X) // 2 :]]
        clf = DecisionTreeClassifier()
        clf = clf.fit(np.array(DT_trainX), np.array(des_lable))
        predictions = []
        with torch.no_grad():
            for arr in Forward_X:
                arr = torch.from_numpy(arr.astype(np.float32))
                pred = DBNmodel(arr)
                class_n = clf.predict(
                    np.array(
                        [
                            float(torch.argmax(pred).cpu().detach().numpy()),
                            float(pred[torch.argmax(pred)].cpu().detach().numpy()),
                        ]
                    ).reshape(1, -1)
                )

                predictions.append(int(class_n))
        Signals["Signal"] = predictions
        signals.append(Signals)

    signals_combained = pd.concat(signals, ignore_index=True, sort=False)
    signals_combained.loc[signals_combained["Signal"] == 0, "Signal"] = -1
    df_stata = bayes_tune_get_stat_after_forward(
        signals_combained,
        patch,
        epochs,
        n_hiden,
        n_hiden_two,
        train_window,
        forward_window,
        source_file_name,
        out_root,
        out_data_root,
        trial.number,
        get_trade_info=True,
    )
    net_profit = df_stata["Net Profit [$]"].values[0]
    Sharpe_Ratio = df_stata["Sharpe Ratio"].values[0]
    trades = df_stata["# Trades"].values[0]
    trial.set_user_attr("# Trades", trades)
    parameters = trial.params
    parameters.update({"trial": trial.number})
    parameters.update({"values_0": net_profit})
    parameters.update({"values_1": Sharpe_Ratio})
    inter = pd.read_excel(
        f"{out_root}/{out_data_root}/intermedia_{source_file_name[:-4]}.xlsx"
    )
    inter = inter.append(parameters, ignore_index=True)
    inter.to_excel(
        f"{out_root}/{out_data_root}/intermedia_{source_file_name[:-4]}.xlsx",
        index=False,
    )

    torch.save(DBNmodel.state_dict(), f"{out_root}/{out_data_root}/weights.pt")

    return net_profit, Sharpe_Ratio


sampler = optuna.samplers.TPESampler(seed=2020)
study = optuna.create_study(directions=["maximize", "maximize"], sampler=sampler)
study.optimize(objective, n_trials=n_trials)


tune_results = study.trials_dataframe()

tune_results["params_forward_window"] = tune_results["params_forward_window"].astype(
    int
)
tune_results["params_train_window"] = tune_results["params_train_window"].astype(int)
df_plot = tune_results[
    [
        "values_0",
        "values_1",
        "user_attrs_# Trades",
        "params_patch",
        "params_epochs",
        "params_n_hiden",
        "params_n_hiden_two",
        "params_train_window",
        "params_forward_window",
    ]
]

fig = px.parallel_coordinates(
    df_plot,
    color="values_0",
    labels={
        "values_0": "Net Profit ($)",
        "values_1": "Sharpe_Ratio",
        "user_attrs_# Trades": "Trades",
        "params_patch": "patch(bars)",
        "params_epochs": "epochs",
        "params_n_hiden": "n_hiden",
        "params_n_hiden_two": "n_hiden_two",
        "params_train_window": "train_window (bars)",
        "params_forward_window": "forward_window (bars)",
    },
    range_color=[df_plot["values_0"].min(), df_plot["values_0"].max()],
    color_continuous_scale=px.colors.sequential.Viridis,
    title=f"bayes_parameters_select_{source_file_name[:-4]}_optune_epoch_{n_trials}",
)

fig.write_html(f"{out_root}/{out_data_root}/hyp_par_sel_{source_file_name[:-4]}.htm")
fig.show()
tune_results.to_excel(
    f"{out_root}/{out_data_root}/hyp_par_sel_{source_file_name[:-4]}.xlsx"
)
