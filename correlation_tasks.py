"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    # Читаем файл. Столбцы разделены табуляцией (sep="\t"),
    # слово NA в файле означает, что значения нет.
    df = pd.read_csv(data.csv_path, sep="\t", na_values=["NA"])

    # Делим таблицу на две части по полу
    men = df[df["Gender"] == "Male"]
    women = df[df["Gender"] == "Female"]

    # Сколько человек в каждой группе
    men_count = len(men)
    women_count = len(women)

    # Признаки, которые сравниваем с MRI_Count
    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]

    women_mri_correlation = {}
    men_mri_correlation = {}

    for feature in features:
        # corr() считает корреляцию Пирсона между двумя столбцами.
        # Строки с пропусками (NaN) pandas пропускает сам.
        women_mri_correlation[feature] = float(women[feature].corr(women["MRI_Count"]))
        men_mri_correlation[feature] = float(men[feature].corr(men["MRI_Count"]))

    # Ищем признак с самым большим модулем корреляции среди обеих групп
    strongest_mri_feature = features[0]
    max_abs_value = 0.0

    for feature in features:
        # abs() нужен, потому что сильная связь бывает и отрицательной
        if abs(women_mri_correlation[feature]) > max_abs_value:
            max_abs_value = abs(women_mri_correlation[feature])
            strongest_mri_feature = feature

        if abs(men_mri_correlation[feature]) > max_abs_value:
            max_abs_value = abs(men_mri_correlation[feature])
            strongest_mri_feature = feature

    # Собираем результат в объект, который требует задание
    result = BrainCorrelationSummary(
        men_count=men_count,
        women_count=women_count,
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=strongest_mri_feature,
    )
    return result