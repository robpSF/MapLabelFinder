import streamlit as st
from collections import Counter
import nltk
from nltk.corpus import stopwords
import re

# Download NLTK stop words if not already available
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Define Streamlit app
def main():
    st.title("Top 100 Most Significant Words")
    
    # Text input for messages
    st.subheader("Input Your Messages")
    messages = st.text_area(
        "Paste your collection of messages below (one per line):",
        placeholder="Enter your messages here..."
    )

    # Process text when the user submits input
    if st.button("Analyze"):
        if messages.strip():
            # Split input into lines and concatenate into one text blob
            text_blob = " ".join(messages.splitlines())

            # Preprocess the text: remove non-alphanumeric characters, tokenize, and filter stop words
            words = preprocess_text(text_blob)

            # Count the word frequencies
            word_counts = Counter(words)

            # Get the top 100 most frequent words
            top_words = word_counts.most_common(100)

            # Display the results
            st.subheader("Top 100 Words")
            for rank, (word, count) in enumerate(top_words, start=1):
                st.write(f"{rank}. {word} ({count} occurrences)")
        else:
            st.error("Please enter some messages to analyze.")

def preprocess_text(text):
    # Remove special characters and digits, keep only words
    cleaned_text = re.sub(r"[^\w\s]", "", text)
    cleaned_text = re.sub(r"\d+", "", cleaned_text)

    # Convert to lowercase and split into words
    words = cleaned_text.lower().split()

    # Remove stop words and short words (less than 2 characters)
    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]

    return filtered_words

if __name__ == "__main__":
    main()
