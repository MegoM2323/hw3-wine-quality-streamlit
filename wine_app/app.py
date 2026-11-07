"""Локальная демонстрация сохранённого пайплайна качества вина."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ui_helpers import (
    FEATURES,
    LABELS,
    batch_predict,
    diagnostic_figure,
    metric_values,
    parse_uploaded_csv,
    validate_features,
)

APP_DIR = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)


@st.cache_resource
def load_model(path: str, version: int) -> Any:
    # Только доверенный локальный артефакт; пользователь загружает лишь CSV.
    return joblib.load(path)


@st.cache_data
def load_test_data(path: str, version: int) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def test_proba(_model: Any, features: pd.DataFrame, model_version: int) -> np.ndarray:
    # Версия файла исключает устаревший результат при замене модели на диске.
    return np.asarray(_model.predict_proba(features))[:, 1]


def read_threshold() -> float:
    with (APP_DIR / "config.json").open(encoding="utf-8") as source:
        value = json.load(source)["threshold"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Порог в config.json должен быть числом.")
    result = float(value)
    if not np.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("Порог в config.json должен находиться между 0 и 1.")
    return result


def single_wine(model: Any, features: pd.DataFrame, threshold: float) -> None:
    st.subheader("Лабораторные характеристики")
    st.caption(
        "Диапазоны и начальные значения взяты из тестовой таблицы. Хорошее вино: экспертная оценка не ниже 7."
    )
    row: dict[str, float] = {}
    columns = st.columns(3)
    for index, name in enumerate(FEATURES[:-1]):
        with columns[index % 3]:
            row[name] = st.number_input(
                LABELS[name],
                min_value=float(features[name].min()),
                max_value=float(features[name].max()),
                value=float(features[name].median()),
                format="%.5f" if name == "density" else "%.3f",
                key=f"wine_{name}",
            )
    row["is_red"] = float(
        st.toggle("Красное вино", value=bool(features["is_red"].median()))
    )
    probability = float(
        model.predict_proba(pd.DataFrame([row], columns=FEATURES))[0, 1]
    )
    st.metric("Вероятность хорошего вина", f"{probability:.1%}")
    if probability >= threshold:
        st.success("Рекомендовать в подборку")
    else:
        st.info("Не включать в подборку при текущем пороге")


def model_metrics(y: pd.Series, probabilities: np.ndarray, threshold: float) -> None:
    st.subheader("Качество на отложенной тестовой выборке")
    values = metric_values(y, probabilities, threshold)
    for column, (name, value) in zip(st.columns(3), values.items()):
        column.metric(name, f"{value:.3f}")
    fig = diagnostic_figure(y, probabilities, threshold)
    st.pyplot(fig)
    plt.close(fig)
    st.write(
        "При увеличении порога подборка сужается: модель требует большей уверенности и пропускает больше хороших вин. При уменьшении — подборка расширяется, но растёт число посредственных вин. Precision на конечной выборке может изменяться немонотонно."
    )
    st.caption(
        "Этот экран показывает эффект порога. Для выбора рабочего порога используйте валидацию, а не тестовую выборку. При отсутствии положительных предсказаний precision принят равным 0."
    )


def batch_tab(model: Any, features: pd.DataFrame, threshold: float) -> None:
    st.subheader("Оценить таблицу вин")
    st.caption(
        "CSV UTF-8 с запятыми: 12 признаков, без target. Порядок столбцов может быть любым."
    )
    st.download_button(
        "Скачать пример входного CSV",
        features.head(10).to_csv(index=False).encode("utf-8"),
        "wine_example.csv",
        "text/csv",
    )
    uploaded = st.file_uploader("CSV с характеристиками вин", type=["csv"])
    if uploaded is None:
        return
    try:
        frame = parse_uploaded_csv(uploaded.getvalue())
        result = batch_predict(model, frame, threshold)
    except (ValueError, TypeError, OverflowError) as exc:
        LOGGER.warning("CSV prediction rejected: %s", exc)
        st.error(str(exc))
        return
    st.dataframe(result, hide_index=True)
    st.download_button(
        "Скачать предсказания",
        result.to_csv(index=False).encode("utf-8"),
        "wine_predictions.csv",
        "text/csv",
    )
    st.caption(f"Строк: {len(result)} · Порог: {threshold:.3f}")


def coefficients_tab(model: Any) -> None:
    coefficients = pd.Series(model.steps[-1][1].coef_[0], index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        [LABELS[name] for name in coefficients.index],
        coefficients,
        color=["#9d2949" if value < 0 else "#287e65" for value in coefficients],
    )
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set(
        xlabel="Коэффициент после StandardScaler",
        title="Вклад признаков в логарифм шансов",
    )
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write(
        f"Сильнее всего за хорошее вино: {LABELS[coefficients.idxmax()]}. Сильнее всего против: {LABELS[coefficients.idxmin()]}."
    )
    st.caption(
        "Коэффициенты относятся к стандартизованным признакам при прочих равных. Коррелирующие признаки влияют на интерпретацию; связь не доказывает причинность."
    )


def main() -> None:
    st.set_page_config(page_title="Винная подборка", page_icon="🍷", layout="wide")
    st.title("🍷 Винная подборка")
    st.write(
        "Проверьте вино, сравните пороги и соберите подборку по лабораторным данным."
    )
    try:
        model_path, test_path = APP_DIR / "model.joblib", APP_DIR / "test.csv"
        model_version = model_path.stat().st_mtime_ns
        model = load_model(str(model_path), model_version)
        test = load_test_data(str(test_path), test_path.stat().st_mtime_ns)
        features = validate_features(test.drop(columns="target"))
        if list(model.feature_names_in_) != FEATURES:
            raise ValueError("Порядок признаков модели не соответствует приложению.")
        y = test["target"]
        if not y.isin([0, 1]).all() or y.nunique() != 2:
            raise ValueError("В test.csv нужны оба класса target: 0 и 1.")
        default_threshold = read_threshold()
        probabilities = test_proba(model, features, model_version)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        EOFError,
        ImportError,
    ) as exc:
        LOGGER.error("Application artifact load failed: %s", exc)
        st.error(
            f"Не удалось загрузить артефакты модели: {exc}. Выполните ноутбук или восстановите файлы model.joblib, config.json и test.csv рядом с app.py."
        )
        st.stop()
    threshold = st.sidebar.slider(
        "Порог рекомендации",
        0.0,
        1.0,
        default_threshold,
        0.001,
        format="%.3f",
        key="threshold",
    )
    st.sidebar.caption(
        "Один порог применяется ко всем вкладкам. Вероятность ≥ порога — рекомендация."
    )
    single, quality, batch, coefficients = st.tabs(
        [
            "Одно вино",
            "Модельные метрики",
            "Пакетное предсказание",
            "Что выучила модель",
        ]
    )
    with single:
        single_wine(model, features, threshold)
    with quality:
        model_metrics(y, probabilities, threshold)
    with batch:
        batch_tab(model, features, threshold)
    with coefficients:
        coefficients_tab(model)


if __name__ == "__main__":
    main()
