"""Графики к части 2: scatter-графики и heatmap корреляций (для отчёта)."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Читаем данные так же, как в analyze_brain_correlations
df = pd.read_csv("brainsize.txt", sep="\t", na_values=["NA"])

# Делим на мужчин и женщин
men = df[df["Gender"] == "Male"]
women = df[df["Gender"] == "Female"]

features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]

# Scatter-графики
# Для каждого признака рисуем свой график: по оси X признак,
# по оси Y MRI_Count. Одна точка это один человек.
for feature in features:
    plt.figure()
    plt.scatter(women[feature], women["MRI_Count"], color="red", label="Женщины")
    plt.scatter(men[feature], men["MRI_Count"], color="blue", label="Мужчины")
    plt.xlabel(feature)
    plt.ylabel("MRI_Count")
    plt.title(f"{feature} и MRI_Count")
    plt.legend()
    plt.savefig(f"scatter_{feature}.png")

# Heatmap корреляций
# corr() считает корреляции между всеми парами столбцов,
# heatmap раскрашивает эту таблицу.
columns = features + ["MRI_Count"]

plt.figure(figsize=(9, 7))
sns.heatmap(women[columns].corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Женщины: корреляции")
plt.savefig("heatmap_women.png")

plt.figure(figsize=(9, 7))
sns.heatmap(men[columns].corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Мужчины: корреляции")
plt.savefig("heatmap_men.png")

# Показываем все окна с графиками
plt.show()