"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

sns.set_theme(style="whitegrid")

df = pd.read_csv('brainsize.txt', sep='\t', na_values='NA')

male = df[df['Gender'] == 'Male']
female = df[df['Gender'] == 'Female']

features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']


fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for i, feature in enumerate(features):
    # Мужчины
    axes[0, i].scatter(male[feature], male['MRI_Count'], alpha=0.7, color='steelblue', edgecolors='black', s=50)
    if male[feature].notna().sum() > 1:
        z = np.polyfit(male[feature].dropna(), male.loc[male[feature].notna(), 'MRI_Count'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(male[feature].min(), male[feature].max(), 100)
        axes[0, i].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=1.5)
    axes[0, i].set_title(f'Male: {feature} vs MRI_Count', fontsize=10)
    axes[0, i].set_xlabel(feature)
    axes[0, i].set_ylabel('MRI_Count')
    
    # Женщины
    axes[1, i].scatter(female[feature], female['MRI_Count'], alpha=0.7, color='salmon', edgecolors='black', s=50)
    if female[feature].notna().sum() > 1:
        z = np.polyfit(female[feature].dropna(), female.loc[female[feature].notna(), 'MRI_Count'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(female[feature].min(), female[feature].max(), 100)
        axes[1, i].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=1.5)
    axes[1, i].set_title(f'Female: {feature} vs MRI_Count', fontsize=10)
    axes[1, i].set_xlabel(feature)
    axes[1, i].set_ylabel('MRI_Count')

plt.suptitle('Зависимость MRI_Count от признаков (scatter + линия тренда)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('scatter_brainsize.png', dpi=150, bbox_inches='tight')
print("Сохранено: scatter_brainsize.png")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Корреляционная матрица для мужчин
male_corr = male[features + ['MRI_Count']].corr()
sns.heatmap(male_corr, annot=True, cmap='coolwarm', center=0, ax=axes[0],
            vmin=-1, vmax=1, fmt='.3f', linewidths=0.5)
axes[0].set_title('Корреляционная матрица (Мужчины)', fontsize=12)

# Корреляционная матрица для женщин
female_corr = female[features + ['MRI_Count']].corr()
sns.heatmap(female_corr, annot=True, cmap='coolwarm', center=0, ax=axes[1],
            vmin=-1, vmax=1, fmt='.3f', linewidths=0.5)
axes[1].set_title('Корреляционная матрица (Женщины)', fontsize=12)

plt.suptitle('Корреляции Пирсона признаков с MRI_Count', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('heatmap_brainsize.png', dpi=150, bbox_inches='tight')
print("Сохранено: heatmap_brainsize.png")
plt.show()