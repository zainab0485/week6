import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def load_dataset(path):
   return pd.read_csv(path)


def explore_dataset(df):
   print(df.head())
   print(df.isnull().sum())
   print(df['label'].value_counts())


def clean_text(text):
   text = text.lower()
   text = re.sub(r"http\S+", "", text)
   text = re.sub(f"[{string.punctuation}]", "", text)
   return text


def preprocess_dataset(df):
   df['clean_text'] = df['text'].apply(clean_text)
   return df


def extract_features(df):
   vectorizer = TfidfVectorizer()
   X = vectorizer.fit_transform(df['clean_text'])
   y = df['label']
   return X, y, vectorizer


def split_data(X, y):
   return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
   model = MultinomialNB()
   model.fit(X_train, y_train)
   return model


def evaluate_model(model, X_test, y_test):
   y_pred = model.predict(X_test)
   print("Accuracy:", accuracy_score(y_test, y_pred))


if __name__ == "__main__":
   df = load_dataset("data.csv")
   explore_dataset(df)

   df = preprocess_dataset(df)

   X, y, vectorizer = extract_features(df)
   X_train, X_test, y_train, y_test = split_data(X, y)

   model = train_model(X_train, y_train)
   evaluate_model(model, X_test, y_test)