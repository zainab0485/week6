import pandas as pd
import pickle

from preprocessing import clean_text
from model import extract_features, split_data, train_model
from evaluation import evaluate_model


def load_dataset(path):
   df = pd.read_csv(path)
   return df


def explore_dataset(df):
   print("First 5 rows:")
   print(df.head())

   print("\nColumns:")
   print(df.columns)

   print("\nMissing values:")
   print(df.isnull().sum())

   print("\nLabel distribution:")
   print(df["label"].value_counts())


def preprocess_dataset(df):
   df["clean_text"] = df["text"].apply(clean_text)
   return df


def save_model(model, vectorizer):
   with open("model.pkl", "wb") as f:
       pickle.dump(model, f)

   with open("vectorizer.pkl", "wb") as f:
       pickle.dump(vectorizer, f)


def load_model():
   with open("model.pkl", "rb") as f:
       model = pickle.load(f)

   with open("vectorizer.pkl", "rb") as f:
       vectorizer = pickle.load(f)

   return model, vectorizer


def predict(text, model, vectorizer):
   cleaned = clean_text(text)
   vector = vectorizer.transform([cleaned])
   pred = model.predict(vector)[0]

   if pred == 0:
       return "negative"
   elif pred == 1:
       return "neutral"
   else:
       return "positive"


if __name__ == "__main__":
   df = load_dataset("sentiment.csv")

   explore_dataset(df)

   df = preprocess_dataset(df)

   X, y, vectorizer = extract_features(df)

   X_train, X_test, y_train, y_test = split_data(X, y)

   nb_model = train_model(X_train, y_train, "naive_bayes")
   print("\nNaive Bayes Results:")
   evaluate_model(nb_model, X_test, y_test)

   lr_model = train_model(X_train, y_train, "logistic_regression")
   print("\nLogistic Regression Results:")
   evaluate_model(lr_model, X_test, y_test)

   save_model(lr_model, vectorizer)

   loaded_model, loaded_vectorizer = load_model()

   print("\nPredictions:")
   print("I love this product ->", predict("I love this product", loaded_model, loaded_vectorizer))
   print("This is bad ->", predict("This is bad", loaded_model, loaded_vectorizer))
   print("It is okay ->", predict("It is okay", loaded_model, loaded_vectorizer))