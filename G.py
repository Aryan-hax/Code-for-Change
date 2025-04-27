from transformers import pipeline
import re

# Function to preprocess text
def preprocess_text(text):
    # Remove special characters and extra spaces
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

# Load sentiment analysis pipeline
credibility_model = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

def check_message_credibility(message):
    preprocessed_message = preprocess_text(message)
    analysis = credibility_model(preprocessed_message)
    
    # Interpret the results
    sentiment = analysis[0]['label']
    score = analysis[0]['score']
    
    if sentiment in ["5 stars", "4 stars"]:
        return f"The message appears credible with a high score of {score:.2f}."
    elif sentiment in ["3 stars"]:
        return f"The message has medium credibility with a score of {score:.2f}."
    else:
        return f"The message seems suspicious with a low score of {score:.2f}."

# Example Usage
unknown_message = input("Enter text message to check credibility: ")
result = check_message_credibility(unknown_message)
print(result)