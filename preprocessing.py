import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def load_dataset(path):
   df = pd.read_csv(path)

   if "text" not in df.columns:
       if "tweet" in df.columns:
           df = df.rename(columns={"tweet": "text"})
       elif "review" in df.columns:
           df = df.rename(columns={"review": "text"})

   if "label" not in df.columns and "sentiment" in df.columns:
       mapping = {"negative": 0, "neutral": 1, "positive": 2}
       df["label"] = df["sentiment"].map(mapping)

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


def clean_text(text):
   text = str(text).lower()
   text = re.sub(r"http\S+", "", text)
   text = re.sub(r"@\w+", "", text)
   text = re.sub(r"[^a-zA-Z\s]", "", text)
   words = text.split()
   words = [word for word in words if word not in ENGLISH_STOP_WORDS]
   return " ".join(words)


def preprocess_dataset(df):
   df["clean_text"] = df["text"].apply(clean_text)
   return df