import pickle
import matplotlib.pyplot as plt

from preprocessing import load_dataset, explore_dataset, preprocess_dataset, clean_text
from model import extract_features, split_data, train_model
from evaluation import evaluate_model


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
   nb_acc = evaluate_model(nb_model, X_test, y_test, "Naive Bayes")

   lr_model = train_model(X_train, y_train, "logistic_regression")
   print("\nLogistic Regression Results:")
   lr_acc = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

   models = ["Naive Bayes", "Logistic Regression"]
   accuracies = [nb_acc, lr_acc]

   plt.bar(models, accuracies)
   plt.title("Model Accuracy Comparison")
   plt.ylabel("Accuracy")
   plt.ylim(0, 1)
   plt.savefig("model_comparison.png")
   plt.close()

   save_model(lr_model, vectorizer)

   loaded_model, loaded_vectorizer = load_model()

   print("\nPredictions:")
   print("I love this product ->", predict("I love this product", loaded_model, loaded_vectorizer))
   print("This is bad ->", predict("This is bad", loaded_model, loaded_vectorizer))
   print("It is okay ->", predict("It is okay", loaded_model, loaded_vectorizer))