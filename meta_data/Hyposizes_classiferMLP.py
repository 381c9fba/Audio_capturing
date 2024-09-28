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
    if pd.isna(text):
        return ''
    tokens = re.findall(r'\w+', text.lower())
    lemmas = [morph.parse(token)[0].normal_form for token in tokens if token not in russian_stopwords and len(token) > 2]
    return ' '.join(lemmas)

def load_data_with_tags(file_path):
    print(f"\nLoading data from {file_path}")
    df = pd.read_csv(file_path)
    print("\nInitial data (first 10 rows):")
    print(df.head(10))
    
    df['tags'] = df['tags'].str.split(';')
    print("\nData after splitting tags (first 10 rows):")
    print(df[['tags']].head(10))

    df['полный_текст'] = df['title'] + " " + df['description']
    df['полный_текст'] = df['полный_текст'].apply(preprocess_text)
    print("\nData after preprocessing full text (first 10 rows):")
    print(df[['полный_текст']].head(10))
    
    return df

def split_main_sub_tags(tags):
    if isinstance(tags, list):
        main_tags = []
        sub_tags = []
        for tag in tags:
            if ':' in tag:
                main_tag, sub_tag = tag.split(':', 1)
                main_tags.append(main_tag.strip())
                sub_tags.append(sub_tag.strip())
            else:
                main_tags.append(tag.strip())
                sub_tags.append(None)
        return main_tags, sub_tags
    return [], []

def binarize_tags(df):
    print("\nBefore processing 'tags':")
    print(df['tags'].head(10))
    df['main_tags'], df['sub_tags'] = zip(*df['tags'].apply(split_main_sub_tags))
    
    print("\nMain and Sub tags (first 10 rows):")
    print(df[['main_tags', 'sub_tags']].head(10))
    
    mlb = MultiLabelBinarizer()
    df_tags_binarized = pd.DataFrame(mlb.fit_transform(df['main_tags']), columns=mlb.classes_, index=df.index)
    
    print("\nBinarized tags (first 10 rows):")
    print(df_tags_binarized.head(10))
    
    print("\nDetected classes (tags):")
    print(mlb.classes_)
    
    return df_tags_binarized, mlb

def train_tags_classifier(X_train, y_train, X_val, y_val):
    print("\nTraining data (first 10 rows):")
    print(X_train[:10])
    print(y_train[:10])

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
    print("\nClassification Report on Validation Set:")
    print(classification_report(y_val, y_val_pred))

    return random_search.best_estimator_

def evaluate_model(classifier, X_test, y_test, mlb):
    y_pred = classifier.predict(X_test)
    
    print("\nPredictions (first 10 rows):")
    print(y_pred[:10])
    
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"\nF1 Score (weighted): {f1}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=mlb.classes_))

def split_data_for_tags(df, df_tags_binarized, test_size=0.2, val_size=0.1):
    print("\nSplitting data into train/val/test sets")
    
    X = df['полный_текст'].tolist()
    y = df_tags_binarized.values
    
    print(f"\nTotal samples: {len(X)}")
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=(test_size + val_size), random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size / (1 - test_size), random_state=42)

    print(f"\nTrain set size: {len(X_train)}, Validation set size: {len(X_val)}, Test set size: {len(X_test)}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def main():
    file_path = 'meta_data/train_data_categories_cleaned.csv'
    df = load_data_with_tags(file_path)
    
    df_tags_binarized, mlb = binarize_tags(df)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data_for_tags(df, df_tags_binarized)
    
    classifier = train_tags_classifier(X_train, y_train, X_val, y_val)

    evaluate_model(classifier, X_test, y_test, mlb)

    joblib.dump({'model': classifier, 'mlb': mlb}, 'tag_classifier_model.joblib')
    logger.info("Model and MultiLabelBinarizer saved")

if __name__ == "__main__":
    main()
