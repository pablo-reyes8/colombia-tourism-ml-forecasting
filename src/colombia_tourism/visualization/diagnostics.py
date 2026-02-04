"""Diagnostic plots for regression models."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_residuals(y_train, y_train_pred, y_test, y_test_pred):
    train_errors = y_train - y_train_pred
    test_errors = y_test - y_test_pred

    train_errors_df = pd.DataFrame(
        {
            "y_train_real": y_train.flatten(),
            "y_train_pred": y_train_pred.flatten(),
            "train_error": train_errors.flatten(),
        }
    )

    test_errors_df = pd.DataFrame(
        {
            "y_test_real": y_test.flatten(),
            "y_test_pred": y_test_pred.flatten(),
            "test_error": test_errors.flatten(),
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=True)

    axes[0].plot(
        train_errors_df.index,
        train_errors_df["train_error"],
        color="#B03060",
        marker="o",
        linestyle="-",
    )
    axes[0].axhline(0, color="black", linestyle="--", linewidth=2)
    axes[0].set_title("Errores de Entrenamiento", fontsize=15, fontweight="bold")
    axes[0].set_xlabel("Numero de observaciones")
    axes[0].set_ylabel("Error")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(
        test_errors_df.index,
        test_errors_df["test_error"],
        color="#FF7F50",
        marker="o",
        linestyle="-",
    )
    axes[1].axhline(0, color="black", linestyle="--", linewidth=2)
    axes[1].set_title("Errores de Prueba", fontsize=15, fontweight="bold")
    axes[1].set_xlabel("Numero de observaciones")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_predictions(y_true, y_pred, title: str = "Predicciones vs Reales"):
    seq = range(len(y_true))
    plt.figure(figsize=(13, 7))
    plt.plot(seq, y_true, label="Reales", linewidth=2, color="#B03060", marker="o")
    plt.plot(
        seq,
        y_pred,
        label="Predicciones",
        linewidth=2,
        color="blue",
        marker="o",
    )
    plt.xlabel("Observaciones")
    plt.ylabel("Turistas esperados")
    plt.title(title)
    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()
