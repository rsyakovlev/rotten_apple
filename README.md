# Классификация порченных яблок

В данном проекте была решена задача классификации порченных яблок с помощью машинного обучения по изображению.


## Данные
Источник данных: https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification


#### Обучающая выборка:
Свежие яблоки: 1693 <br>
Порченные яблоки: 2342 <br>

#### Тестовая выборка:
Свежие яблоки: 395 <br>
Порченные яблоки: 601 <br>


## Ход работы
Ход работы и выбор финальной модели показаны в блокноте ./notebook/Rotten Apples Classification Task.ipynb.

Изначально были протестированы классические алгоритмы машинного обучения. <br>
Для получения предикторов были рассчитаны статистические признаки на основе трёхканального тензора изображения. <br>
Были протестированы Логистическая регрессия, SVM, K-Nearest Neighbors, Random Forest, Catboost, Light GBM, XGBoost. <br>
Для каждой модели были подобраны по Grid Search на кросс-валидации гиперпараметры. <br>
С помощью ансамблей деревьев и K-Nearest Neighbors удалось получить Accuracy и F1-score ~98%. <br>
Повысить качество модели удалось с помощью стэкинга всех построенных моделей.


## Обучение
Обучение модели производится с помощью скрипта train.py.
Аргументы:
* -s (default=224)
* -f (default="apples")
* -b (default=8)
* -l (default=0.0001)
* -i (default=5)
* -e (default=10)
* -m (default="models/my_model")

Пример использования в shell:
```
python train.py
```


## Валидация
Валидация модели производится с помощью скрипта validate.py.
Аргументы:
* -m (default="models/my_model")
* -f (default="apples/test")
* -s (default=224)
* -b (default=32)
* -e (default=1)

Пример использования в shell:
```
python validate.py -m "models/my_model" -f "apples/test"
```


## Предсказание
Предсказание моделью производится с помощью скрипта validate.py. Сделать предсказание можно для изображения или для директории с изображениями.
Аргументы:
* -m (default="./models/my_model")
* -i (default="./apples/examples")
* -s (default=224)
* -e (default=1)

Пример использования в shell:
```
python predict.py
```


## Web-приложение:

Пример использования в shell:
```
python app/app.py
```

Пример использования в Docker:
```
docker build -t rotten_apple .
docker run -it -p 5000:5000 rotten_apple
```
