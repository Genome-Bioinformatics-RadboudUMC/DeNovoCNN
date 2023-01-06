"""
evaluate.py

Copyright (c) 2023 Karolis Sablauskas
Copyright (c) 2023 Gelana Khazeeva

This file is part of DeNovoCNN.

DeNovoCNN is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

DeNovoCNN is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Foobar.  If not, see <https://www.gnu.org/licenses/>.
"""


import sklearn.metrics as metrics
from denovonet.training.utils import apply_model_images_onxx
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import numpy.typing as npt
import pandas as pd


def calculate_metrics_for_paper(
    y_test: npt.NDArray, y_prob: npt.NDArray, tag: str
) -> pd.DataFrame:
    y_pred = (np.array(y_prob) > 0.5).astype(int)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)

    # save confusion matrix and slice into four pieces
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]

    # calculate the specificity
    conf_specificity = TN / float(TN + FP)

    metrics_dict = {}
    metrics_dict["ROC AUC"] = round(metrics.roc_auc_score(y_test, y_prob), 4)
    metrics_dict["Accuracy"] = round(metrics.accuracy_score(y_test, y_pred), 4)
    metrics_dict["Sensitivity"] = round(metrics.recall_score(y_test, y_pred), 4)
    metrics_dict["Specificity"] = round(conf_specificity, 4)
    metrics_dict["F1 score"] = round(metrics.f1_score(y_test, y_pred), 4)
    metrics_dict["Precision"] = round(metrics.precision_score(y_test, y_pred), 4)
    metrics_dict["True Positives"] = TP
    metrics_dict["False Positives"] = FP
    metrics_dict["True Negatives"] = TN
    metrics_dict["False Negatives"] = FN

    return pd.DataFrame(
        data=metrics_dict.values(), index=metrics_dict.keys(), columns=[tag]
    ).transpose()


def plot_roc_curve(
    y_true: npt.NDArray, y_pred: npt.NDArray, tag: str, ax: Axes
) -> None:
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    ax.plot(fpr, tpr, label=f"{tag} network: AUC = %0.2f" % roc_auc)


def plot_precision_recall_curve(
    y_true: npt.NDArray, y_pred: npt.NDArray, tag: str, ax: Axes, style: str
) -> None:
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
    pr_auc = metrics.auc(recall, precision)

    ax.plot(
        recall,
        precision,
        style,
        label=f"{tag} network: PR-AUC = %0.2f" % pr_auc,
        linewidth=1.8,
    )


def evaluate_models(
    images_folder: str,
    model_sub_path: str,
    model_in_path: str,
    model_del_path: str,
    metrics_out_path: str,
    figure_out_path: str,
) -> None:
    images_sub, target_sub, predictions_sub = apply_model_images_onxx(
        model_path=model_sub_path,
        images_folder=os.path.join(images_folder, "substitution", "test"),
    )

    images_ins, target_ins, predictions_ins = apply_model_images_onxx(
        model_path=model_in_path,
        images_folder=os.path.join(images_folder, "insertion", "test"),
    )

    images_del, target_del, predictions_del = apply_model_images_onxx(
        model_path=model_del_path,
        images_folder=os.path.join(images_folder, "deletion", "test"),
    )

    metrics_all = calculate_metrics_for_paper(
        y_test=target_sub + target_ins + target_del,
        y_prob=predictions_sub + predictions_ins + predictions_del,
        tag="Total",
    )

    metrics_sub = calculate_metrics_for_paper(
        y_test=target_sub, y_prob=predictions_sub, tag="Substitution"
    )

    metrics_ins = calculate_metrics_for_paper(
        y_test=target_ins, y_prob=predictions_ins, tag="Insertion"
    )

    metrics_del = calculate_metrics_for_paper(
        y_test=target_del, y_prob=predictions_del, tag="Deletion"
    )

    metrics_all = pd.concat(
        [metrics_all, metrics_sub, metrics_ins, metrics_del], axis=0
    )
    metrics_all = (
        metrics_all.transpose().reset_index().rename(columns={"index": "Metric"})
    )
    metrics_all.to_csv(metrics_out_path, sep="\t", index=False)

    # Plot AUC-ROC curve
    sns.set_style("whitegrid")
    sns.set_context("paper")

    fig, ax = plt.subplots(figsize=(10, 6))

    # method I: plt
    plt.title("PR-AUC", fontsize=16)
    plot_precision_recall_curve(
        target_sub, predictions_sub, tag="Substitutions", ax=ax, style="-r"
    )
    plot_precision_recall_curve(
        target_ins, predictions_ins, tag="Insertions", ax=ax, style="-g"
    )
    plot_precision_recall_curve(
        target_del, predictions_del, tag="Deletions", ax=ax, style="-b"
    )

    plt.legend(loc="lower left", fontsize=14)
    ax.set_ylim(0.2, 1.05)
    ax.grid(linestyle="--", linewidth=0.5)
    ax.tick_params(labelsize=10)

    plt.ylabel("Precision", fontsize=14)
    plt.xlabel("Recall", fontsize=14)
    plt.savefig(figure_out_path)
