import pandas as pd
import re
import nltk
import string
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """
    Cleans input text by removing special characters, stopwords, and lemmatizing.
    
    Parameters:
        text (str): Input text string.
        
    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove special characters, punctuation, and digits
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize text
    words = word_tokenize(text)

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

def preprocess_texts(texts, num_words=10000, maxlen=100):
    """Tokenize and pad sequences for model input."""
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post')
    return tokenizer, padded_sequences

def save_tokenizer(tokenizer, filepath="tokenizer.pkl"):
    """Save tokenizer for future use."""
    with open(filepath, "wb") as f:
        pickle.dump(tokenizer, f)

if __name__ == "__main__":
    # data = pd.read_csv("../data/IMDB Dataset.csv")
    # tokenizer, processed_data = preprocess_texts(data['review'])
    # save_tokenizer(tokenizer)
    # print("Preprocessing complete. Tokenizer saved.")
    data='A wonderful little production. <br /><br />'
    # preprocess_texts(data)
    print(clean_text(data))
