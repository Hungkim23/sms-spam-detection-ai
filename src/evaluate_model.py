import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
import pandas as pd
from collections import Counter
import numpy as np

def save_outputs(fig, filename):
    os.makedirs("outputs", exist_ok=True)
    fig.savefig(os.path.join("outputs", filename), bbox_inches='tight')
    plt.close(fig)

def evaluate_model(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    
    return acc, prec, rec, f1, roc

def plot_eda(df, cleaned_texts):
    os.makedirs("outputs", exist_ok=True)
    
    # a. spam_distribution.png
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='label', hue='label', palette='Set2', legend=False, ax=ax)
    ax.set_title("Spam vs Ham Distribution")
    save_outputs(fig, "spam_distribution.png")
    
    # b. message_length_distribution.png
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='message_length', hue='label', bins=50, kde=True, ax=ax)
    ax.set_title("Message Length Distribution")
    save_outputs(fig, "message_length_distribution.png")
    
    # c & d. top_spam_words.png and top_ham_words.png
    spam_texts = [text for text, label in zip(cleaned_texts, df['label_num']) if label == 1]
    ham_texts = [text for text, label in zip(cleaned_texts, df['label_num']) if label == 0]
    
    def get_top_words(texts, n=20):
        all_words = " ".join(texts).split()
        return Counter(all_words).most_common(n)
        
    spam_top = get_top_words(spam_texts)
    ham_top = get_top_words(ham_texts)
    
    if spam_top:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=[w[1] for w in spam_top], y=[w[0] for w in spam_top], hue=[w[0] for w in spam_top], palette="Reds_r", legend=False, ax=ax)
        ax.set_title("Top 20 Spam Words")
        save_outputs(fig, "top_spam_words.png")
        
    if ham_top:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=[w[1] for w in ham_top], y=[w[0] for w in ham_top], hue=[w[0] for w in ham_top], palette="Blues_r", legend=False, ax=ax)
        ax.set_title("Top 20 Ham Words")
        save_outputs(fig, "top_ham_words.png")

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    save_outputs(fig, "confusion_matrix.png")

def plot_roc_curve(y_true, y_prob):
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        save_outputs(fig, "roc_curve.png")
