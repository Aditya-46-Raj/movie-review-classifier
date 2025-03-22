import pandas as pd

def load_data(filepath="../data/IMDB Dataset.csv"):
    """Load IMDB dataset from CSV."""
    return pd.read_csv(filepath)

if __name__ == "__main__":
    data = load_data()
    print("Dataset Shape:", data.shape)
