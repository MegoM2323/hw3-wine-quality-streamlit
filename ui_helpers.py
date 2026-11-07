"""Проверка входных таблиц и вычисления для демонстрации модели."""

from __future__ import annotations

import csv
import io
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn import metrics

FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "is_red",
]
LABELS = {
    "fixed_acidity": "Фиксированная кислотность",
    "volatile_acidity": "Летучая кислотность",
    "citric_acid": "Лимонная кислота",
    "residual_sugar": "Остаточный сахар",
    "chlorides": "Хлориды",
    "free_sulfur_dioxide": "Свободный диоксид серы",
    "total_sulfur_dioxide": "Общий диоксид серы",
    "density": "Плотность",
    "pH": "pH",
    "sulphates": "Сульфаты",
    "alcohol": "Алкоголь",
    "is_red": "Красное вино",
}


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Порядок колонок не влияет на соответствие коэффициентам модели."""
    if frame.empty:
        raise ValueError("В CSV нет строк с данными.")
    if frame.columns.duplicated().any():
        raise ValueError("В CSV повторяются названия столбцов.")
    missing = sorted(set(FEATURES) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(FEATURES))
    if missing:
        raise ValueError("Не хватает столбцов: " + ", ".join(missing))
    if extra:
        raise ValueError(
            "Лишние столбцы (включая target) нужно удалить: " + ", ".join(extra)
        )
    try:
        result = (
            frame.loc[:, FEATURES].apply(pd.to_numeric, errors="raise").astype(float)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Все признаки должны содержать только числа.") from exc
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError(
            "Обнаружены пропуски или бесконечные значения. Заполните их конечными числами."
        )
    if not result["is_red"].isin([0, 1]).all():
        raise ValueError("Признак is_red должен быть 0 (белое) или 1 (красное).")
    return result


def parse_uploaded_csv(content: bytes) -> pd.DataFrame:
    try:
        text = content.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text), strict=True)
        header = next(reader, [])
        if not header:
            raise ValueError("CSV-файл пуст.")
        if len(header) != len(set(header)):
            raise ValueError("В CSV повторяются названия столбцов.")
        # pandas иначе может молча принять первое лишнее поле за индекс строки.
        for row in reader:
            if row and len(row) != len(header):
                raise ValueError(
                    f"В строке CSV {reader.line_num} неверное число полей: "
                    f"ожидалось {len(header)}, получено {len(row)}."
                )
        frame = pd.read_csv(io.StringIO(text))
    except UnicodeDecodeError as exc:
        raise ValueError("Сохраните CSV в кодировке UTF-8.") from exc
    except (pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error) as exc:
        raise ValueError(
            "Не удалось прочитать CSV. Используйте запятые как разделитель."
        ) from exc
    return validate_features(frame)


def batch_predict(model: Any, frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    result = validate_features(frame)
    proba = np.asarray(model.predict_proba(result))[:, 1]
    result["probability"] = proba
    result["prediction"] = (proba >= threshold).astype(int)
    return result


def metric_values(
    y: pd.Series, proba: np.ndarray, threshold: float
) -> dict[str, float]:
    predicted = (proba >= threshold).astype(int)
    return {
        "Precision": float(metrics.precision_score(y, predicted, zero_division=0)),
        "Recall": float(metrics.recall_score(y, predicted, zero_division=0)),
        "F1": float(metrics.f1_score(y, predicted, zero_division=0)),
    }


def diagnostic_figure(y: pd.Series, proba: np.ndarray, threshold: float) -> Figure:
    predicted = (proba >= threshold).astype(int)
    values = metric_values(y, proba, threshold)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics.ConfusionMatrixDisplay.from_predictions(
        y, predicted, labels=[0, 1], ax=axes[0], colorbar=False
    )
    axes[0].set(title="Матрица ошибок", xlabel="Предсказание", ylabel="Истинный класс")
    fpr, tpr, _ = metrics.roc_curve(y, proba)
    tn, fp, fn, tp = metrics.confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    axes[1].plot(fpr, tpr, label=f"ROC-AUC = {metrics.roc_auc_score(y, proba):.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].scatter(
        [fp / (fp + tn)],
        [tp / (tp + fn)],
        color="crimson",
        label=f"Порог {threshold:.3f}",
    )
    axes[1].set(
        title="ROC-кривая",
        xlabel="False positive rate",
        ylabel="Recall",
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
    )
    precision, recall, _ = metrics.precision_recall_curve(y, proba)
    axes[2].plot(
        recall, precision, label=f"AP = {metrics.average_precision_score(y, proba):.3f}"
    )
    axes[2].axhline(
        float(y.mean()), linestyle="--", color="gray", label="Случайная модель"
    )
    axes[2].scatter(
        [values["Recall"]],
        [values["Precision"]],
        color="crimson",
        label=f"Порог {threshold:.3f}",
    )
    axes[2].set(
        title="PR-кривая",
        xlabel="Recall",
        ylabel="Precision",
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
    )
    axes[1].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    return fig
