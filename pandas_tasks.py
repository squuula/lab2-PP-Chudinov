"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    # Читаем csv-файл в таблицу
    df = pd.read_csv(data.csv_path)

    # Сколько всего строк в таблице
    row_count = len(df)

    # Считаем пропуски в каждом столбце.
    # isna() даёт True там, где значения нет, sum() считает эти True.
    missing_by_column = {}
    for column in df.columns:
        missing_by_column[column] = int(df[column].isna().sum())

    # Сколько пассажиров старше 30 лет.
    adults_over_30_count = int((df["Age"] > 30).sum())

    # Средний возраст и доля выживших для каждого класса.
    mean_age_by_pclass = {}
    survival_rate_by_pclass = {}

    for pclass in [1, 2, 3]:
        # Берём только пассажиров нужного класса
        one_class = df[df["Pclass"] == pclass]

        # mean() - ср. арифм столбца, сам пропускает NaN
        mean_age_by_pclass[pclass] = float(one_class["Age"].mean())

        # В Survived только 0 и 1, поэтому среднее равно доле выживших
        survival_rate_by_pclass[pclass] = float(one_class["Survived"].mean())

    # Пять самых больших тарифов: сортируем по убыванию и берём первые 5
    sorted_fares = df["Fare"].sort_values(ascending=False)
    highest_fares = sorted_fares.head(5).tolist()

    # Собираем всё в объект, который требует задание
    result = TitanicSummary(
        row_count=row_count,
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares,
    )
    return result