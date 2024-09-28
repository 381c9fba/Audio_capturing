import logging
import re
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import nltk
import pymorphy3
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set the model name and initialize the tokenizer
MODEL_NAME = 'roberta-base'  # You can use 'roberta-large', 'xlm-roberta-base', etc.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
morph = pymorphy3.MorphAnalyzer()

nltk.download('stopwords')
russian_stopwords = nltk.corpus.stopwords.words('russian')

def preprocess_text(text):
    """
    Функция для предобработки текста. Преобразует текст в нижний регистр, убирает стоп-слова и применяет лемматизацию.
    """
    if pd.isna(text):
        return ''
    tokens = re.findall(r'\w+', text.lower())
    lemmas = [morph.parse(token)[0].normal_form for token in tokens if token not in russian_stopwords and len(token) > 2]
    return ' '.join(lemmas)

def load_data_with_tags(file_path):
    """
    Функция для загрузки данных и обработки тегов.
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
    """
    df['main_tags'], df['sub_tags'] = zip(*df['tags'].apply(split_main_sub_tags))
    
    print("\nОсновные и под-теги (первые 10 строк):")
    print(df[['main_tags', 'sub_tags']].head(10))
    
    mlb_main = MultiLabelBinarizer()
    df_main_tags_binarized = pd.DataFrame(mlb_main.fit_transform(df['main_tags']), columns=mlb_main.classes_, index=df.index)
    
    print("\nБинаризованные основные теги (первые 10 строк):")
    print(df_main_tags_binarized.head(10))
    
    return df_main_tags_binarized, mlb_main

def tokenize_function(examples):
    """
    Функция для токенизации текстов с использованием токенизатора RoBERTa.
    """
    return tokenizer(examples['полный_текст'], padding="max_length", truncation=True, max_length=512)

def train_main_tag_classifier(X_train, y_train, X_val, y_val, model_name, device):
    """
    Функция для обучения RoBERTa классификатора для основных тегов.
    """
    train_dataset = Dataset.from_pandas(X_train)
    val_dataset = Dataset.from_pandas(X_val)

    # Токенизация данных
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Преобразование в тензоры
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Модель для классификации
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=y_train.shape[1])

    # Настройки обучения
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True
    )

    # Обучение модели
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    trainer.train()

    return model

def evaluate_model(model, tokenizer, X_test, device):
    """
    Функция для оценки обученной модели на тестовых данных.
    """
    test_dataset = Dataset.from_pandas(X_test)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    trainer = Trainer(model=model, tokenizer=tokenizer)
    
    predictions = trainer.predict(test_dataset)
    return predictions

def main():
    """
    Основная функция для загрузки данных, обучения классификаторов и сохранения моделей.
    """
    file_path = 'meta_data/train_data_categories_cleaned.csv'
    df = load_data_with_tags(file_path)
    
    df_main_tags_binarized, mlb_main = binarize_main_tags(df)

    X_train, X_test, y_train, y_test = train_test_split(df[['полный_текст']], df_main_tags_binarized, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main_tag_model = train_main_tag_classifier(X_train, y_train, X_val, y_val, MODEL_NAME, device)

    y_pred = evaluate_model(main_tag_model, tokenizer, X_test, device)

    print(f"\nПредсказания (первые 10 строк):\n{y_pred[:10]}")

    joblib.dump({'main_tag_classifier': main_tag_model, 'mlb_main': mlb_main}, 'tag_classifiers_roberta.joblib')

if __name__ == "__main__":
    main()
