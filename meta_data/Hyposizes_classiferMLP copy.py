import logging
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
import pymorphy3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

morph = pymorphy3.MorphAnalyzer()

nltk.download('stopwords')
russian_stopwords = nltk.corpus.stopwords.words('russian')

def preprocess_text(text):
    """
    Функция для предобработки текста. 
    Преобразует текст в нижний регистр, убирает стоп-слова и применяет лемматизацию.
    
    Аргументы:
    text -- текст для предобработки
    
    Возвращает:
    Лемматизированный и очищенный текст.
    """
    if pd.isna(text):
        return ''
    tokens = re.findall(r'\w+', text.lower())
    lemmas = [morph.parse(token)[0].normal_form for token in tokens if token not in russian_stopwords and len(token) > 2]
    return ' '.join(lemmas)

def load_data_with_tags(file_path):
    """
    Функция для загрузки данных и обработки тегов.
    
    Аргументы:
    file_path -- путь к CSV-файлу с данными
    
    Возвращает:
    DataFrame с обработанными тегами и текстом.
    """
    print(f"\nЗагрузка данных из {file_path}")
    df = pd.read_csv(file_path)
    print("\nИсходные данные (первые 10 строк):")
    print(df.head(10))
    
    df['tags'] = df['tags'].str.split(';')
    print("\nДанные после разделения тегов (первые 10 строк):")
    print(df[['tags']].head(10))

    df['полный_текст'] = df['title'] + " " + df['description']
    df['полный_текст'] = df['полный_текст'].apply(preprocess_text)
    print("\nДанные после предобработки текста (первые 10 строк):")
    print(df[['полный_текст']].head(10))
    
    return df

def split_main_sub_tags(tags):
    """
    Функция для разделения тегов на основные и под-теги.
    
    Аргументы:
    tags -- список тегов
    
    Возвращает:
    Два списка: основных тегов и под-тегов. Если под-тег отсутствует, используется 'No Sub-Tag'.
    """
    main_tags = []
    sub_tags = []
    if isinstance(tags, list):
        for tag in tags:
            if ':' in tag:
                main_tag, sub_tag = tag.split(':', 1)
                main_tags.append(main_tag.strip())
                sub_tags.append(sub_tag.strip())
            else:
                main_tags.append(tag.strip())
                sub_tags.append("No Sub-Tag")  # Заполнитель для отсутствующего под-тега
    return main_tags, sub_tags

def binarize_main_tags(df):
    """
    Функция для бинаризации основных тегов.
    
    Аргументы:
    df -- DataFrame с данными и тегами
    
    Возвращает:
    Бинаризованные основные теги и объект MultiLabelBinarizer для основных тегов.
    """
    df['main_tags'], df['sub_tags'] = zip(*df['tags'].apply(split_main_sub_tags))
    
    print("\nОсновные и под-теги (первые 10 строк):")
    print(df[['main_tags', 'sub_tags']].head(10))
    
    mlb_main = MultiLabelBinarizer()
    df_main_tags_binarized = pd.DataFrame(mlb_main.fit_transform(df['main_tags']), columns=mlb_main.classes_, index=df.index)
    
    print("\nБинаризованные основные теги (первые 10 строк):")
    print(df_main_tags_binarized.head(10))
    
    return df_main_tags_binarized, mlb_main

def binarize_sub_tags(df):
    """
    Функция для бинаризации под-тегов.
    
    Аргументы:
    df -- DataFrame с данными и тегами
    
    Возвращает:
    Бинаризованные под-теги и объект MultiLabelBinarizer для под-тегов.
    """
    mlb_sub = MultiLabelBinarizer()
    df_sub_tags_binarized = pd.DataFrame(mlb_sub.fit_transform(df['sub_tags']), columns=mlb_sub.classes_, index=df.index)
    
    print("\nБинаризованные под-теги (первые 10 строк):")
    print(df_sub_tags_binarized.head(10))
    
    return df_sub_tags_binarized, mlb_sub

def train_main_tag_classifier(X_train, y_train, X_val, y_val):
    """
    Функция для обучения классификатора для основных тегов.
    
    Аргументы:
    X_train -- обучающие данные (текст)
    y_train -- обучающие метки (основные теги)
    X_val -- данные для валидации
    y_val -- метки для валидации
    
    Возвращает:
    Обученный классификатор для основных тегов.
    """
    pipeline = Pipeline(
        [
            ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 4))),
            ('clf', OneVsRestClassifier(MLPClassifier(random_state=42, max_iter=500)))
        ]
    )

    param_dist = {
        'clf__estimator__hidden_layer_sizes': [(100,), (50, 50)],
        'clf__estimator__activation': ['relu'],
        'clf__estimator__solver': ['adam'],
        'clf__estimator__alpha': [0.0001, 0.001],
        'clf__estimator__learning_rate': ['adaptive'],
        'clf__estimator__learning_rate_init': [0.001]
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=100,
        cv=kf,
        random_state=42,
        n_jobs=-1,
        scoring="f1_weighted",
        verbose=2
    )
    random_search.fit(X_train, y_train)
    best_pipeline = random_search.best_estimator_

    y_val_pred = best_pipeline.predict(X_val)
    print("\nОтчет о классификации для основных тегов на валидационном наборе:")
    print(classification_report(y_val, y_val_pred))

    return random_search.best_estimator_

def train_sub_tag_classifier(main_tag_preds, y_sub_tags):
    """
    Функция для обучения классификатора для под-тегов.
    
    Аргументы:
    main_tag_preds -- предсказанные основные теги
    y_sub_tags -- метки под-тегов
    
    Возвращает:
    Обученный классификатор для под-тегов.
    """
    pipeline = Pipeline(
        [
            ('clf', OneVsRestClassifier(MLPClassifier(random_state=42, max_iter=500)))
        ]
    )

    pipeline.fit(main_tag_preds, y_sub_tags)
    return pipeline

def evaluate_model(classifier, X_test, y_test, mlb):
    """
    Функция для оценки модели на тестовых данных.
    
    Аргументы:
    classifier -- обученная модель
    X_test -- тестовые данные (предсказанные теги)
    y_test -- истинные метки
    mlb -- объект MultiLabelBinarizer для восстановления исходных тегов
    
    Возвращает:
    Отчет о классификации и F1-метрика.
    """
    y_pred = classifier.predict(X_test)
    
    print("\nПредсказания (первые 10 строк):")
    print(y_pred[:10])
    
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"\nF1-метрика (взвешенная): {f1}")
    print("\nОтчет о классификации:")
    print(classification_report(y_test, y_pred, target_names=mlb.classes_))

def main():
    """
    Основная функция для загрузки данных, обучения классификаторов и сохранения моделей.
    """
    file_path = 'meta_data/train_data_categories_cleaned.csv'
    df = load_data_with_tags(file_path)
    
    df_main_tags_binarized, mlb_main = binarize_main_tags(df)
    df_sub_tags_binarized, mlb_sub = binarize_sub_tags(df)

    (X_train, y_main_train), (X_val, y_main_val), (X_test, y_main_test) = split_main_sub_tags(df, df_main_tags_binarized)
    
    main_tag_classifier = train_main_tag_classifier(X_train, y_main_train, X_val, y_main_val)

    y_main_pred = main_tag_classifier.predict(X_test)

    sub_tag_classifier = train_sub_tag_classifier(y_main_pred, df_sub_tags_binarized.values)

    evaluate_model(sub_tag_classifier, y_main_pred, df_sub_tags_binarized.values, mlb_sub)

    joblib.dump({'main_tag_classifier': main_tag_classifier, 'sub_tag_classifier': sub_tag_classifier, 'mlb_main': mlb_main, 'mlb_sub': mlb_sub}, 'tag_classifiers.joblib')
    logger.info("Классификаторы для основных и под-тегов сохранены")

if __name__ == "__main__":
    main()