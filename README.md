# Product-sentiment-analysis-Russian-Lang

## Description
Русско-язычная модель сентимент анализа отзывов на товары.
Модель состоит из:
1. Трансформер `TfidfVectorizer`
2. Классификатор `LogisticRegressionCV`
Качество на кросс-валидации по метрике `accuracy` - **86%**.
Веб-форма для демонстрации работы алгоритма разработана при помощи
фреймворка `Flask`.

## How it works ?
1. Создать `venv` 
1. `python demo.py`
2. Перейти по адресу `http://127.0.0.1:8080/sentiment`

## Requirements
• `Python` - 3.7.3
• `Flask` - 1.1.2
• `Scikit-learn` - 0.23.2
• `joblib` - 0.16