import pandas as pd
import string
import re

def clean_text(text):
    """
    Cleans text by converting to lowercase, removing punctuation,
    and removing extra spaces. Handles non-string inputs.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Replace numbers with token "number" instead of completely removing them
    text = re.sub(r'\d+', 'number', text)
    
    # Remove extra spaces
    text = " ".join(text.split())
    
    return text
