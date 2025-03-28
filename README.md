# 🎬 Movie Review Classifier

A Sentiment Analysis System for Movie Reviews using Machine Learning & Deep Learning

## 📌 Overview
This project is a **Movie Review Classifier** that analyzes user reviews and classifies them as **positive** or **negative** using **Machine Learning** and **Deep Learning** techniques.

### Implemented Models:
- **Linear SVC**
- **Random Forest Classifier**
- **Neural Network**
- **Logistic Regression**

The dataset is **preprocessed**, **vectorized**, and used for **training** models. The best-performing model is selected based on accuracy and precision.

## 🚀 Features
✅ **Preprocessing Pipeline** – Cleans and vectorizes text data.  
✅ **Multiple Machine Learning Models** – Compare different classifiers.  
✅ **Deep Learning Integration** – Neural Network model for better accuracy.  
✅ **Model Saving & Loading** – Store trained models for future use.  
✅ **Performance Evaluation** – Accuracy & precision comparison.  

## 📁 Project Structure
```
📂 movie_review_classifier/
│── 📂 data/                    # Dataset & Processed Data
│   ├── raw/                    # Raw dataset
│   ├── processed/               # Preprocessed & vectorized data
│   ├── X_train.csv              # Training feature data
│   ├── X_test.csv               # Testing feature data
│   ├── Y_train.csv              # Training labels
│   ├── Y_test.csv               # Testing labels
│
│── 📂 models/                   # Saved Trained Models
│   ├── saved_models/            # Pre-trained models
│   ├── linear_svc.pkl           # Linear SVC model
│   ├── random_forest.pkl        # Random Forest model
│   ├── logistic_regression.pkl  # Logistic Regression model
│   ├── Neural_Net.pkl           # Neural Network model
│   ├── tokenizer.pkl            # Tokenizer for text vectorization
│
│── 📂 results/                   # Evaluation Metrics
│   ├── plots/                    # Model Performance Visualizations
│   ├── confusion_matrix_svc.png  
│   ├── confusion_matrix_rf.png  
│   ├── model_accuracy_comparison.png  
│
│── 📂 notebooks/                 # Jupyter Notebooks for Analysis
│   ├── Movie_review_classifier.ipynb  # Main notebook
│   ├── test.ipynb                 # Testing notebook
│
│── 📂 src/                        # Source Code
│   ├── data_loader.py             # Handles dataset loading
│   ├── preprocess.py              # Text preprocessing pipeline
│   ├── model.py                   # Model training & evaluation
│   ├── utils.py                   # Helper functions
│
│── 📂 venv/                        # Virtual Environment (if used)
│── .gitignore                      # Ignore unnecessary files
│── README.md                       # Project Documentation
│── requirements.txt                 # Dependencies
```

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/movie_review_classifier.git
cd movie_review_classifier
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Train Models
```bash
python src/model.py
```

### 5️⃣ Evaluate Models
```bash
python src/utils.py
```

## 📊 Model Performance
| Model               | Accuracy | Precision |
|---------------------|----------|-----------|
| **Linear SVC**      | 51.31%    | 51.36%     |
| **Random Forest**   | 53.88%    | 53.94%     |
| **Neural Network**  | 88.05%    | 88.05%     |
| **Logistic Regression** | 51.31% | 51.36%    |
![alt text](results/plots/model_performance_comparison.png)

## 🎯 Future Improvements
- 🔍 Try **XGBoost** or **Gradient Boosting** for better accuracy.
- 🎯 Implement **Hyperparameter tuning**.
- 🌐 **Deploy the model** as a web application.

## 📜 License
This project is licensed under the **[MIT License](LICENSE)** - see the LICENSE file for details.

## 🤝 Contributing
Contributions are welcome! Feel free to **fork** this repository and submit a **pull request**.

## 📞 Contact
For any queries, reach out to:\
📧 Email: adityaraj21103@gmail.com  
🐦 Twitter: [Aditya_Raj_0211](https://twitter.com/Aditya_Raj_0211)  
📌 GitHub: [Aditya-46-Raj](https://github.com/Aditya-46-Raj)

