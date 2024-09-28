import numpy as np
import pandas as pd
import matplotlib
import os
import sys
from collections import Counter
import matplotlib.pyplot as plt

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

def preprocess_data(data, reference_columns=None):
    mapping = {
        'Укажите Ваш пол': {'Мужчина': 0, 'Женщина': 1},
        'Какой напиток Вы предпочитаете утром?': {'Чай': 0, 'Кофе': 1},
        'Курите ли Вы?': {'Да': 1, 'Нет': 0},
        'Вы высыпаетесь?': {'Да': 1, 'Нет': 0},
        'Укажите Ваш хронотип': {'Сова': 0, 'Жаворонок': 1},
        'Если поблизости с Вашим домом кофейня?': {'Да': 1, 'Нет': 0},
        'Вы являетесь гурманом?': {'Да': 1, 'Нет': 0},
        'Вы работаете из офиса?': {'Да': 1, 'Нет': 0},
        'Вы домосед?': {'Да': 1, 'Нет': 0},
        'У Вас есть хронические заболевания?': {'Да': 1, 'Нет': 0},
        'Какой рукой Вы пишите?': {'Левой': 0, 'Правой': 1},
        'Какой у Вас знак зодиака?': {
            'Овен': 0, 'Телец': 1, 'Близнецы': 2, 'Рак': 3, 'Лев': 4,
            'Дева': 5, 'Весы': 6, 'Скорпион': 7, 'Стрелец': 8,
            'Козерог': 9, 'Водолей': 10, 'Рыбы': 11
        },
        'Укажите цвет Вашего левого глаза': {
            'Зеленый': 0, 'Карий': 1, 'Голубой': 2, 'Серо-зеленый': 3,
            'Серый': 4, 'Серо-голубой': 5, 'Синий': 6, 'Коричневый': 7
        }
    }

    for column in mapping:
        if column in data.columns:
            data[column] = data[column].map(mapping[column])

    data = data.fillna(0)
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

    if reference_columns is not None:
        for col in reference_columns:
            if col not in data.columns:
                data[col] = 0
        data = data[reference_columns]

    return data

file_path = 'Результаты-опроса.csv'
df = pd.read_csv(file_path)

features = df.drop(columns=['Какой напиток Вы предпочитаете утром?'])
labels = df['Какой напиток Вы предпочитаете утром?']

X = preprocess_data(features).values
y = labels.values

knn = KNN(k=3)
knn.fit(X, y)

test_data = pd.read_csv('test_data.csv')

X_test = preprocess_data(test_data, reference_columns=features.columns).values

predictions = knn.predict(X_test)
print(predictions)

def is_running_in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

if is_running_in_colab():
    matplotlib.use('module://matplotlib_inline.backend_inline')
else:
    if os.environ.get('DISPLAY') is None and sys.platform != 'win32':
        matplotlib.use('Agg')  
    else:
        try:
            import PyQt5
            matplotlib.use('Qt5Agg')  
        except ImportError:
            matplotlib.use('TkAgg')  )

drink_preference_count = df['Какой напиток Вы предпочитаете утром?'].value_counts()

plt.figure(figsize=(10, 6))
plt.bar(drink_preference_count.index, drink_preference_count.values, alpha=0.7)
plt.xlabel('Напиток (0 - Чай, 1 - Кофе)')
plt.ylabel('Частота')
plt.title('Частота выбора напитка утром')

if matplotlib.get_backend() == 'Agg':
    plt.savefig('output_drink_preference.png')
else:
    plt.show()

plt.close()  

factor_columns = df.columns.drop('Какой напиток Вы предпочитаете утром?')

for column in factor_columns:
    if df[column].nunique() <= 10:  
        factor_preference = df.groupby([column, 'Какой напиток Вы предпочитаете утром?']).size().unstack()
        factor_preference.plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.title(f'Влияние "{column}" на выбор напитка')
        plt.xlabel(column)
        plt.ylabel('Количество')
        plt.legend(title='Напиток', labels=['Чай', 'Кофе'])
        
        if matplotlib.get_backend() == 'Agg':
            plt.savefig(f'output_{column}.png')
        else:
            plt.show()
        
        plt.close()  
