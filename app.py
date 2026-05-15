import streamlit as st
import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.utils import load_data, prepare_dataset

st.set_page_config(page_title="SMS Spam Detection AI", page_icon="📩", layout="wide")

st.title("📩 SMS Spam Detection AI Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["Dataset Overview", "Text EDA", "Model Comparison", "Predict Message"])

# Helper function to load data
@st.cache_data
def get_data():
    try:
        df = load_data("data/spam.csv")
        df = prepare_dataset(df)
        return df
    except Exception as e:
        return None

df = get_data()

with tab1:
    st.header("Dataset Overview")
    if df is not None:
        st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
        st.dataframe(df.head())
        
        spam_count = (df['label_num'] == 1).sum()
        ham_count = (df['label_num'] == 0).sum()
        spam_ratio = spam_count / len(df) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Ham Messages", ham_count)
        col2.metric("Spam Messages", spam_count)
        col3.metric("Spam Ratio", f"{spam_ratio:.2f}%")
        
        st.write(f"**Missing values:** {df.isnull().sum().sum()}")
        st.write("**Note:** Duplicates have been removed during dataset preparation.")
    else:
        st.error("Dataset not found. Please place spam.csv inside the data/ folder.")

with tab2:
    st.header("Text Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("outputs/spam_distribution.png"):
            st.image("outputs/spam_distribution.png", use_container_width=True)
            st.write("Distribution of Spam and Ham messages in the dataset.")
        else:
            st.warning("spam_distribution.png not found.")
            
        if os.path.exists("outputs/top_spam_words.png"):
            st.image("outputs/top_spam_words.png", use_container_width=True)
            st.write("Top 20 most frequent words in Spam messages.")
        else:
            st.warning("top_spam_words.png not found.")
            
    with col2:
        if os.path.exists("outputs/message_length_distribution.png"):
            st.image("outputs/message_length_distribution.png", use_container_width=True)
            st.write("Comparison of message lengths between Spam and Ham.")
        else:
            st.warning("message_length_distribution.png not found.")
            
        if os.path.exists("outputs/top_ham_words.png"):
            st.image("outputs/top_ham_words.png", use_container_width=True)
            st.write("Top 20 most frequent words in Ham messages.")
        else:
            st.warning("top_ham_words.png not found.")

with tab3:
    st.header("Model Comparison")
    
    if os.path.exists("models/model_metrics.csv"):
        metrics_df = pd.read_csv("models/model_metrics.csv")
        # Highlight best model based on F1-score
        st.dataframe(metrics_df.style.highlight_max(subset=['F1-score'], color='lightgreen'))
        
        st.markdown("""
        **Metrics Explanation:**
        - **Accuracy**: Ratio of correct predictions.
        - **Precision**: How many predicted spams are actually spam. (High precision = fewer false alarms).
        - **Recall**: How many actual spams were detected. (High recall = catches more spam).
        - **F1-score**: Harmonic mean of Precision and Recall. Best metric for balancing the trade-off.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists("outputs/confusion_matrix.png"):
                st.image("outputs/confusion_matrix.png", use_container_width=True)
        with col2:
            if os.path.exists("outputs/roc_curve.png"):
                st.image("outputs/roc_curve.png", use_container_width=True)
    else:
        st.warning("Model metrics not found. Please run `python src/train_model.py` first.")

with tab4:
    st.header("Predict Message")
    
    if not os.path.exists("models/best_spam_model.pkl"):
        st.warning("Model not found! Please run `python src/train_model.py` to train the models.")
    else:
        model = joblib.load("models/best_spam_model.pkl")
        
        user_input = st.text_area("Enter a message to classify:")
        
        if st.button("Predict"):
            if user_input.strip() == "":
                st.warning("Please enter a message before prediction.")
            else:
                # Predicting
                pred = model.predict([user_input])[0]
                
                # Try getting probability
                prob = None
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba([user_input])[0][1]
                elif hasattr(model, "decision_function"):
                    score = model.decision_function([user_input])[0]
                    import numpy as np
                    prob = 1 / (1 + np.exp(-score))
                
                if pred == 1:
                    st.error("SPAM")
                    st.warning("This message is likely to be spam. Be careful with suspicious links, prizes, urgent payment requests, or unknown senders.")
                else:
                    st.success("NORMAL MESSAGE")
                    st.success("This message looks like a normal message.")
                
                if prob is not None:
                    st.write(f"**Spam Probability:** {prob:.2f}")
                    
                    if prob < 0.3:
                        st.info("Risk level: Low Risk")
                    elif 0.3 <= prob < 0.7:
                        st.warning("Risk level: Medium Risk")
                    else:
                        st.error("Risk level: High Risk")

