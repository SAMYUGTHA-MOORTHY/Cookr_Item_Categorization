from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import ast 
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import Tk, Label, Entry, Button, ttk

# Load dataset
data = pd.read_csv("C:/Users/csdhi/Desktop/Cookr/Random Forest/Food_train dataset.csv")

# Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

# Preprocess item names
data['Item Name'] = data['Item Name'].apply(preprocess_text)

# Extract unique categories
unique_categories = []
for categories_list in data['Categories'].apply(ast.literal_eval):
    unique_categories.extend(categories_list)

# Convert to a set to ensure uniqueness, and then back to a list to make it hashable
unique_categories = list(set(unique_categories))

# Split data into features and target
X = data['Item Name']
y = [ast.literal_eval(categories) for categories in data['Categories']]

# Encode categories using unique categories
mlb = MultiLabelBinarizer(classes=unique_categories)
y_encoded = mlb.fit_transform(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build model pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer(analyzer='word', tokenizer=word_tokenize, preprocessor=preprocess_text)),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])


# Train model
model.fit(X_train, y_train)

# Predict categories for test data
predictions = model.predict(X_test)

# Decode predicted categories
decoded_predictions = mlb.inverse_transform(predictions)
from tkinter import Tk, Label, Entry, ttk

# Function to handle user input and display predictions
def get_user_input():
    user_input = entry.get()
    
    # Preprocess the user input
    preprocessed_user_input = preprocess_text(user_input)
    print(preprocessed_user_input)

    # Predict categories using the trained model
    user_predictions = model.predict([preprocessed_user_input])

    # Decode predicted categories
    decoded_user_predictions = mlb.inverse_transform(user_predictions)

    # Display the decoded categories for the user input
    output_label.config(text=f"{user_input.lower()}: {decoded_user_predictions}")

# Create a themed Tkinter window
window = Tk()
window.title("Category Prediction")

# Set window dimensions
window_width = 600
window_height = 400
window.geometry(f"{window_width}x{window_height}")

# Entry widget for user input
entry_label = Label(window, text="Enter text:")
entry_label.pack(pady=10)
entry = Entry(window, font=("Arial", 12))
entry.pack(pady=10)

# Configure a style for the button
style = ttk.Style()
style.configure("TButton", padding=5, relief="flat", background="#4CAF50", font=("Arial", 12))

# Button to trigger prediction
predict_button = ttk.Button(window, text="Predict", command=get_user_input, style="TButton")
predict_button.pack(pady=10)

# Label to display output
output_label = Label(window, text="", font=("Arial", 12))
output_label.pack(pady=10)

# Run the Tkinter event loop
window.mainloop()
