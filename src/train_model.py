import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from utils import load_data, prepare_dataset
from preprocessing import clean_text
from evaluate_model import evaluate_model, plot_eda, plot_confusion_matrix, plot_roc_curve

def train_models():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    print("Loading data...")
    try:
        df = load_data("data/spam.csv")
    except FileNotFoundError as e:
        print(e)
        return

    print("Preparing dataset...")
    df = prepare_dataset(df)
    
    print("Cleaning text...")
    # Apply cleaning
    df['cleaned_message'] = df['message'].apply(clean_text)
    
    print("Generating EDA plots...")
    plot_eda(df, df['cleaned_message'].tolist())
    
    X = df['cleaned_message']
    y = df['label_num']
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": SVC(kernel='linear', probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier()
    }
    
    metrics_list = []
    best_f1 = -1
    best_model_name = ""
    best_pipeline = None
    best_y_pred = None
    best_y_prob = None
    
    print("Training models...")
    for name, model in models.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
            ("model", model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        y_prob = None
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            y_prob = pipeline.decision_function(X_test)
            # Normalize to 0-1 for ROC AUC if needed
            if y_prob.max() != y_prob.min():
                y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
            else:
                y_prob = None
            
        acc, prec, rec, f1, roc = evaluate_model(y_test, y_pred, y_prob)
        
        metrics_list.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1,
            "ROC-AUC": roc
        })
        
        print(f"{name} trained. F1-score: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_pipeline = pipeline
            best_y_pred = y_pred
            best_y_prob = y_prob
            
    print(f"\nBest model: {best_model_name} with F1-score: {best_f1:.4f}")
    
    # Save outputs
    os.makedirs("models", exist_ok=True)
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv("models/model_metrics.csv", index=False)
    
    joblib.dump(best_pipeline, "models/best_spam_model.pkl")
    
    print("Generating evaluation plots for best model...")
    plot_confusion_matrix(y_test, best_y_pred)
    if best_y_prob is not None:
        plot_roc_curve(y_test, best_y_prob)
        
    print("Training complete! Files saved to outputs/ and models/")

if __name__ == "__main__":
    train_models()
