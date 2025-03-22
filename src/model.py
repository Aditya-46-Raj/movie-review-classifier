from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input
import pickle
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

def create_model(input_length, vocab_size, embedding_dim=128):
    """Create an LSTM model for sentiment analysis."""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=input_length),
        LSTM(128, return_sequences=True),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=5, batch_size=32):
    """Train the LSTM model."""
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
    return model

def save_model(model, filepath="../models/saved_models/",name = "model.pkl"):
    """Save the trained model."""
    filepath = filepath + name
    joblib.dump(model, filepath)

# if __name__ == "__main__":
#     with open("tokenizer.pkl", "rb") as f:
#         tokenizer = pickle.load(f)
    
#     data = np.load("processed_data.npy", allow_pickle=True)
#     labels = np.load("labels.npy")
    
#     X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
    
#     model = create_model(input_length=100, vocab_size=10000)
#     model = train_model(model, X_train, y_train, X_val, y_val)
#     save_model(model)
#     print("Model training complete. Model saved.")
