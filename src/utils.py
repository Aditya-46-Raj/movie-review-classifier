import numpy as np
import pickle
from tensorflow.keras.models import load_model

def load_tokenizer(filepath="tokenizer.pkl"):
    """Load a saved tokenizer."""
    with open(filepath, "rb") as f:
        return pickle.load(f)

def load_trained_model(filepath="lstm_model.h5"):
    """Load a trained LSTM model."""
    return load_model(filepath)

def preprocess_new_texts(texts, tokenizer, maxlen=100):
    """Tokenize and pad new texts for prediction."""
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post')
    return padded_sequences

def predict_sentiment(model, texts, tokenizer):
    """Predict sentiment for new texts."""
    processed_texts = preprocess_new_texts(texts, tokenizer)
    predictions = model.predict(processed_texts)
    return ["Positive" if p > 0.5 else "Negative" for p in predictions]

if __name__ == "__main__":
    tokenizer = load_tokenizer()
    model = load_trained_model()
    sample_texts = ["This movie was amazing!", "I did not like the plot."]
    predictions = predict_sentiment(model, sample_texts, tokenizer)
    print(predictions)
# Output: ['Positive', 'Negative']
# This script loads a trained model and tokenizer, preprocesses new texts, and predicts their sentiment.