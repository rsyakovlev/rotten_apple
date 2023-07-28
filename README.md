# Классификация порченных яблок

В данном проекте была решена задача классификации порченных яблок с помощью машинного обучения.<br>
Источник данных: https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification <br>
Ход работы и выбор финальной модели показаны в блокноте ./notebook/Rotten Apples Classification Task.ipynb. <br>

## Обучение
Обучение производится с помощью скрипта train.py.
Аргументы:
@click.option('-s', '--img_size', default=224)
@click.option('-f', '--data_folder', default="apples")
@click.option('-b', '--train_batch_size', default=8)
@click.option('-l', '--lr', default=0.0001)
@click.option('-i', '--sch_total_iters', default=5)
@click.option('-e', '--epochs', default=10)
@click.option('-m', '--model_dir', default="models/my_model")

Пример:
```
python train.py
```


## Валидация
Аргументы:
* -m (--model_dir). default="models/my_model"
* -f (--data_dir). default="apples/test"
* -s (--img_size). default=224
* -b (--batch_size). default=32
* -e (--export_to_file). default=1

Пример:
```
python validate.py -m "models/my_model" -f "apples/test"
```

## Предсказание
Сделать предсказание для изображения или директории с изображениями
```
python predict.py
```


Запустить приложение:

cmd

python app/app.py

Dockerfile
