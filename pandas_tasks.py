"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    text = pd.read_csv('titanic.csv')
    row_count = len(text)
    missing_by_column = text.isnull().sum().to_dict()
    adults_over_30_count = len(text[text['Age'] > 30])
    mean_age_by_pclass_n = text.groupby('Pclass')['Age'].mean()
    mean_age_by_pclass = {int(k): float(v) for k, v in mean_age_by_pclass_n.to_dict().items()}
    survival_rate_by_pclass_n = text.groupby('Pclass')['Survived'].count() / row_count
    survival_rate_by_pclass = {int(k): float(v) for k, v in survival_rate_by_pclass_n.to_dict().items()}
    highest_fares = text['Fare'].nlargest(5).tolist()
    return(
        row_count,
        missing_by_column,
        adults_over_30_count,
        mean_age_by_pclass,
        survival_rate_by_pclass,
        highest_fares
    )

result = analyze_titanic(TitanicInput(csv_path='titanic.csv'))
print(f"Количество строк          : {result.row_count}")
print(f"Пропуски по столбцам      : {result.missing_by_column}")
print(f"Пассажиров старше 30 лет  : {result.adults_over_30_count}")
print(f"Средний возраст по классам: {result.mean_age_by_pclass}")
print(f"Доля выживших по классам  : {result.survival_rate_by_pclass}")
print(f"Топ-5 тарифов (Fare)      : {result.highest_fares}")