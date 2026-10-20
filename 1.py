
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# Створення трьох груп студентів
groups = [
    ([85, 90, 7, 18, 9], 34),
    ([65, 75, 5, 12, 6], 33),
    ([45, 60, 2, 7, 3], 33)
]

# Випадкові відхилення для кожної ознаки
spread = [4, 4, 0.7, 1.5, 0.8]

objects = []

for center, count in groups:
    group = rng.normal(center, spread, size=(count, 5))
    objects.append(group)

# Об'єднуємо всі групи
X = np.vstack(objects)

# Обмеження допустимих значень
X = np.rint(X).astype(int)

minimum = [0, 0, 0, 0, 0]
maximum = [100, 100, 24, 20, 10]

X = np.clip(X, minimum, maximum)

# Створення таблиці
data = pd.DataFrame(X, columns=[
    "Оцінка",
    "Відвідуваність",
    "Години навчання",
    "Виконані завдання",
    "Активність"
])

# Зберігаємо 100 об'єктів
data.to_csv("data.csv", index=False, encoding="utf-8-sig")

print("Кількість об'єктів:", len(data))
print("\nПерші 10 об'єктів:")
print(data.head(10))





# Завдання 2. Субтрактивна кластеризація

# Нормалізація даних
X = data.to_numpy(dtype=float)
minimum = X.min(axis=0)
maximum = X.max(axis=0)
X_norm = (X - minimum) / (maximum - minimum)

# Параметри алгоритму
radius = 0.5
squash = 1.25
accept_ratio = 0.5
reject_ratio = 0.15

alpha = 4 / radius**2
beta = 4 / (squash * radius)**2

# Обчислення відстаней між об'єктами
distances = np.sum(
    (X_norm[:, None, :] - X_norm[None, :, :])**2,
    axis=2
)

# Обчислення потенціалів
potential = np.exp(-alpha * distances).sum(axis=1)
first_potential = potential.max()

centers = []

# Пошук центрів кластерів
for _ in range(100):
    index = np.argmax(potential)
    peak = potential[index]

    if peak < reject_ratio * first_potential:
        break

    candidate = X_norm[index].copy()
    accepted = peak >= accept_ratio * first_potential

    if not accepted and centers:
        nearest = min(
            np.linalg.norm(candidate - c)
            for c in centers
        )

        accepted = (
            nearest / radius + peak / first_potential
        ) >= 1

    if accepted:
        centers.append(candidate)

        distance_to_center = np.sum(
            (X_norm - candidate)**2,
            axis=1
        )

        potential -= peak * np.exp(
            -beta * distance_to_center
        )
        potential = np.maximum(potential, 0)
    else:
        potential[index] = 0

centers = np.array(centers)

# Повернення до початкових значень
centers_original = (
    centers * (maximum - minimum) + minimum
)

# Створення таблиці центрів
result = pd.DataFrame(
    centers_original,
    columns=data.columns
)

# Сортування за оцінкою
result = result.sort_values(
    by="Оцінка"
).reset_index(drop=True)

# Назви груп
group_names = [
    "Низька успішність",
    "Середня успішність",
    "Висока успішність"
]

if len(result) == 3:
    result.insert(0, "Назва групи", group_names)
else:
    result.insert(
        0,
        "Назва групи",
        [f"Група {i+1}" for i in range(len(result))]
    )

# Збереження центрів
result.to_csv(
    "centers_subtractive.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nСубтрактивна кластеризація")
print("Знайдено кластерів:", len(result))
print(result.round(2).to_string(index=False))

# Побудова графіка
plt.figure(figsize=(11, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    color="skyblue",
    alpha=0.7,
    label="Студенти"
)

plt.scatter(
    result["Оцінка"],
    result["Відвідуваність"],
    color="red",
    marker="X",
    s=200,
    label="Центри кластерів"
)

# Підписи центрів
for i in range(len(result)):
    plt.annotate(
        result.loc[i, "Назва групи"],
        (
            result.loc[i, "Оцінка"],
            result.loc[i, "Відвідуваність"]
        ),
        xytext=(7, 8),
        textcoords="offset points",
        fontsize=9
    )

plt.xlabel("Оцінка")
plt.ylabel("Відвідуваність")
plt.title("Субтрактивна кластеризація студентів")
plt.legend()
plt.grid(True)
plt.xlim(30, 110)
plt.ylim(45, 105)
plt.tight_layout()

plt.savefig("subtractive.png", dpi=150)
plt.show()








# Завдання 2. Кластеризація методом Fuzzy C-Means

import skfuzzy as fuzz

# Використовуємо ті самі нормалізовані дані
# X_norm отримано в попередньому завданні

# Кластеризація на 3 групи
centers_fcm, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    X_norm.T,
    c=3,
    m=2,
    error=0.005,
    maxiter=1000,
    seed=42
)

# Повертаємо центри до початкового масштабу
centers_original_fcm = (
    centers_fcm * (maximum - minimum) + minimum
)

# Таблиця центрів кластерів
result_fcm = pd.DataFrame(
    centers_original_fcm,
    columns=data.columns
)

# Сортуємо центри за оцінкою
result_fcm = result_fcm.sort_values(
    by="Оцінка"
).reset_index(drop=True)

# Назви кластерів
group_names = [
    "Низька успішність",
    "Середня успішність",
    "Висока успішність"
]

result_fcm.insert(0, "Назва групи", group_names)

# Збереження центрів
result_fcm.to_csv(
    "centers_fcm.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nКластеризація методом Fuzzy C-Means")
print("Кількість кластерів:", len(result_fcm))
print(result_fcm.round(2).to_string(index=False))

# Визначаємо основний кластер кожного студента
labels = np.argmax(u, axis=0)

# Перенумерація кластерів за зростанням оцінки
order = np.argsort(centers_original_fcm[:, 0])
labels = np.array([
    np.where(order == label)[0][0]
    for label in labels
])

# Графік
plt.figure(figsize=(11, 6))

colors = ["orange", "skyblue", "lightgreen"]

for i in range(3):
    plt.scatter(
        X[labels == i, 0],
        X[labels == i, 1],
        color=colors[i],
        alpha=0.7,
        label=group_names[i]
    )

# Центри кластерів
plt.scatter(
    result_fcm["Оцінка"],
    result_fcm["Відвідуваність"],
    color="red",
    marker="X",
    s=220,
    label="Центри кластерів"
)

# Підписи центрів
for i in range(3):
    plt.annotate(
        result_fcm.loc[i, "Назва групи"],
        (
            result_fcm.loc[i, "Оцінка"],
            result_fcm.loc[i, "Відвідуваність"]
        ),
        xytext=(7, 8),
        textcoords="offset points",
        fontsize=9
    )

plt.xlabel("Оцінка")
plt.ylabel("Відвідуваність")
plt.title("Кластеризація студентів методом Fuzzy C-Means")
plt.legend()
plt.grid(True)
plt.xlim(30, 110)
plt.ylim(45, 105)
plt.tight_layout()

plt.savefig("fuzzy_c_means.png", dpi=150)
plt.show()



# Приклад із 4 групами


# Додатковий експеримент: Fuzzy C-Means із 4 кластерами

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Використовуємо дані зі 100 студентами
X4 = data.to_numpy(dtype=float)

# Нормалізація даних
min4 = X4.min(axis=0)
max4 = X4.max(axis=0)
X_norm4 = (X4 - min4) / (max4 - min4)

# Кластеризація на 4 групи
centers4, u4, _, _, _, _, _ = fuzz.cluster.cmeans(
    X_norm4.T,
    c=4,
    m=2,
    error=0.005,
    maxiter=1000,
    seed=42
)

# Повертаємо центри до початкового масштабу
centers_original4 = centers4 * (max4 - min4) + min4

# Сортуємо кластери за оцінкою
order4 = np.argsort(centers_original4[:, 0])
centers_original4 = centers_original4[order4]

# Назви чотирьох груп
group_names4 = [
    "Низька успішність",
    "Нижча за середню",
    "Вища за середню",
    "Висока успішність"
]

# Створюємо таблицю центрів
result4 = pd.DataFrame(
    centers_original4,
    columns=data.columns
)

result4.insert(0, "Назва групи", group_names4)

# Зберігаємо центри
result4.to_csv(
    "centers_fcm_4.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nFuzzy C-Means із 4 кластерами")
print("Кількість кластерів:", len(result4))
print(result4.round(2).to_string(index=False))

# Визначаємо основний кластер кожного студента
labels4 = np.argmax(u4, axis=0)

# Перенумеровуємо кластери після сортування
new_labels4 = np.zeros_like(labels4)

for new_id, old_id in enumerate(order4):
    new_labels4[labels4 == old_id] = new_id

# Побудова графіка
plt.figure(figsize=(12, 6))

colors4 = ["orange", "gold", "skyblue", "lightgreen"]

for i in range(4):
    plt.scatter(
        X4[new_labels4 == i, 0],
        X4[new_labels4 == i, 1],
        color=colors4[i],
        alpha=0.7,
        label=group_names4[i]
    )

# Центри кластерів
plt.scatter(
    result4["Оцінка"],
    result4["Відвідуваність"],
    color="red",
    marker="X",
    s=220,
    label="Центри кластерів"
)

# Підписи центрів
for i in range(4):
    plt.annotate(
        result4.loc[i, "Назва групи"],
        (
            result4.loc[i, "Оцінка"],
            result4.loc[i, "Відвідуваність"]
        ),
        xytext=(7, 8),
        textcoords="offset points",
        fontsize=9
    )

plt.xlabel("Оцінка")
plt.ylabel("Відвідуваність")
plt.title("Кластеризація Fuzzy C-Means (4 кластери)")
plt.legend()
plt.grid(True)
plt.xlim(30, 110)
plt.ylim(45, 105)
plt.tight_layout()

# Збереження графіка
plt.savefig("fuzzy_c_means_4.png", dpi=150)
plt.show()
