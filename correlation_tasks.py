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
    text = pd.read_csv(data.csv_path, sep='\t', na_values='NA')
    text_male = text[text['Gender'] == 'Male']
    text_female = text[text['Gender'] == 'Female']
    men_count = int(len(text_male))
    women_count = int(len(text_female))
    
    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']
    
    men_mri_correlation = text_male[features].corrwith(text_male['MRI_Count']).to_dict()
    men_mri_correlation = {k: float(v) for k, v in men_mri_correlation.items()}
    
    women_mri_correlation = text_female[features].corrwith(text_female['MRI_Count']).to_dict()
    women_mri_correlation = {k: float(v) for k, v in women_mri_correlation.items()}

    strongest_mri_feature = ''
    max_abs_corr = -1.0
    
    for feature in features:
        m_corr = abs(men_mri_correlation.get(feature, 0.0))
        w_corr = abs(women_mri_correlation.get(feature, 0.0))
        
        if m_corr > max_abs_corr:
            max_abs_corr = m_corr
            strongest_mri_feature = feature
        if w_corr > max_abs_corr:
            max_abs_corr = w_corr
            strongest_mri_feature = feature
            
    return BrainCorrelationSummary(
        men_count=men_count,
        women_count=women_count,
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=strongest_mri_feature,
    )

result = analyze_brain_correlations(BrainDataInput(csv_path='brainsize.txt'))
print(f"Количество мужчин: {result.men_count}")
print(f"Количество женщин: {result.women_count}")
print(f"Корреляции у мужчин: {result.men_mri_correlation}")
print(f"Корреляции у женщин: {result.women_mri_correlation}")
print(f"Самый сильный признак: {result.strongest_mri_feature}")