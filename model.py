from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


def extract_features(df):
   vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
   X = vectorizer.fit_transform(df["clean_text"])
   y = df["label"]
   return X, y, vectorizer


def split_data(X, y):
   return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_model(X_train, y_train, model_name="naive_bayes"):
   if model_name == "naive_bayes":
       model = MultinomialNB()
   elif model_name == "logistic_regression":
       model = LogisticRegression(max_iter=1000)
   else:
       raise ValueError("Model name must be 'naive_bayes' or 'logistic_regression'")

   model.fit(X_train, y_train)
   return model