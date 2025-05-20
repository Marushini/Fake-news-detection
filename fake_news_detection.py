import nltk
# Download necessary NLTK resources
nltk.download('punkt')       # Download the punkt resource
nltk.download('punkt_tab')   # Download the missing punkt_tab resource
nltk.download('stopwords')   # Download stopwords

import pandas as pd                     # For dataset loading and manipulation
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.feature_extraction.text import TfidfVectorizer  # For text feature extraction (TF-IDF)
from sklearn.ensemble import RandomForestClassifier  # For the machine learning model
from sklearn.metrics import classification_report    # For evaluation metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt         # For plotting visualizations (ROC, confusion matrix)
import seaborn as sns                  # For better heatmap visualization of confusion matrix
import pickle                            # For saving/loading the model and vectorizer
import os                                # For managing file paths and checking directories
from flask import Flask, request, jsonify  # For Flask API

# Debugging: Check the current working directory
print("Current working directory:", os.getcwd())

# Preprocessing function
def preprocess_text(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))  # Stopwords from NLTK
    tokens = nltk.word_tokenize(text.lower())  # Tokenize and convert to lowercase
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

# Check if 'news.csv' exists in the current working directory
file_path = os.path.join(os.getcwd(), 'news.csv')

if not os.path.exists(file_path):
    print(f"Error: 'news.csv' file not found. Please ensure it is located in the directory: {os.getcwd()}")
    exit()

# Load dataset
try:
    data = pd.read_csv(file_path)  # Loading 'news.csv' from the current directory
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: Could not load the file at {file_path}. Make sure it's named 'news.csv'.")
    exit()

# Ensure the dataset contains the necessary columns
if 'text' not in data.columns or 'label' not in data.columns:
    print("Error: Dataset must contain 'text' and 'label' columns.")
    exit()

# Preprocess the text
print("Preprocessing text...")
data['cleaned_text'] = data['text'].apply(preprocess_text)

# Split the dataset into training and testing sets
X = data['cleaned_text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to TF-IDF features
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Random Forest model
print("Training the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# Evaluate the model
print("Evaluating the model...")
y_pred = model.predict(X_test_tfidf)

# Classification report for precision, recall, and F1-score
print("Classification Report:\n", classification_report(y_test, y_pred))

# ROC-AUC Score
y_prob = model.predict_proba(X_test_tfidf)[:, 1]
roc_auc = auc(*roc_curve(y_test, y_prob)[:2])
print(f"ROC-AUC: {roc_auc}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
plt.plot(thresholds, precision[:-1], color='blue', label='Precision')
plt.plot(thresholds, recall[:-1], color='orange', label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Save the model and vectorizer for future use
print("Saving the model and vectorizer...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Function for making predictions
def predict_news(text):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)[0]
    return "Fake News" if prediction == 1 else "Real News"

# Example usage
print("\nPredicting on new text:")
example_text = "Aliens have landed in New York City and taken over Times Square!"
prediction = predict_news(example_text)
print(f"Text: {example_text}\nPrediction: {prediction}")

# Flask API for real-time prediction
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)[0]
    result = "Fake News" if prediction == 1 else "Real News"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
