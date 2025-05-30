# src/ai_models/nlp_processor.py
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import Counter
from typing import List, Dict, Tuple
import pandas as pd # Added for preprocess_text_series

def download_nltk_resources():
    """Downloads necessary NLTK resources (stopwords, punkt) if not already present."""
    try:
        stopwords.words('english')
        print("NLTK stopwords already available.")
    except LookupError:
        print("NLTK stopwords not found. Downloading...")
        nltk.download('stopwords')
        print("NLTK stopwords downloaded.")
    try:
        word_tokenize("test") # A simple way to check if 'punkt' is available
        print("NLTK punkt tokenizer already available.")
    except LookupError:
        print("NLTK punkt tokenizer not found. Downloading...")
        nltk.download('punkt')
        print("NLTK punkt tokenizer downloaded.")

# Call the function to ensure resources are available when the module is imported.
# download_nltk_resources() # We might call this explicitly from app startup instead.

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> List[str]:
    """Cleans and preprocesses text: lowercase, remove punctuation, tokenize, remove stopwords, stem."""
    if not isinstance(text, str):
        print(f"Warning: preprocess_text received non-string input: {type(text)}. Returning empty list.")
        return []
    # Lowercase
    text = text.lower()
    # Remove punctuation (and numbers, for simplicity here)
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and stem
    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and word.isalpha()]
    # print(f"Preprocessing text: '{text[:50]}...' -> Tokens: {processed_tokens[:10]}...")
    return processed_tokens

def preprocess_text_series(texts: pd.Series) -> pd.Series:
    """Applies preprocess_text to each element of a Pandas Series."""
    if not isinstance(texts, pd.Series):
        print(f"Warning: preprocess_text_series received non-Series input: {type(texts)}. Returning empty Series.")
        return pd.Series([], dtype=object)
    return texts.apply(preprocess_text)

def process_text_data(text: str) -> List[str]:
    """Processes text data by cleaning and preprocessing.
       Currently a wrapper around preprocess_text.
    """
    # This function seems to be expected by other modules.
    # For now, it will just call preprocess_text.
    # Further investigation might be needed to confirm its exact original purpose.
    print(f"process_text_data called with: {text[:50]}...")
    return preprocess_text(text)

def analyze_sentiment_vader(text_tokens: List[str]) -> Dict[str, float]:
    """VADER-like sentiment analysis. Returns a dictionary with positive, negative, neutral, compound scores."""
    # This is a very simplified placeholder. Real VADER or other libraries would be much more sophisticated.
    positive_keywords = ["good", "great", "happi", "posit", "love", "like", "excel", "benefit"]
    negative_keywords = ["bad", "problem", "negat", "hate", "dislik", "terribl", "issu", "concern"]

    score = 0
    num_meaningful_words = 0

    for token in text_tokens:
        if token in positive_keywords:
            score += 1
            num_meaningful_words +=1
        elif token in negative_keywords:
            score -= 1
            num_meaningful_words +=1
        elif len(token) > 2: # count other words that are not stop words and are stemmed
            num_meaningful_words +=1

    compound = 0
    if num_meaningful_words > 0:
        compound = score / num_meaningful_words
        # Normalize to -1 to 1 range (crude normalization)
        compound = max(-1, min(1, compound))

    # Crude positive/negative/neutral scores based on compound
    pos_score = max(0, compound)
    neg_score = abs(min(0, compound))
    neu_score = 1 - (pos_score + neg_score)
    neu_score = max(0, min(1, neu_score)) # ensure it's within [0,1]

    # print(f"Analyzing sentiment for tokens: {text_tokens[:10]}... Score: {compound}")
    return {'neg': neg_score, 'neu': neu_score, 'pos': pos_score, 'compound': compound}

def batch_analyze_sentiment(texts_tokens: List[List[str]]) -> List[Dict[str, float]]:
    """Analyzes a batch of tokenized texts for sentiment using analyze_sentiment_vader.
    Expects a list of documents, where each document is a list of tokens.
    Returns a list of sentiment dictionaries.
    """
    sentiments = []
    if not isinstance(texts_tokens, list) or not all(isinstance(doc, list) for doc in texts_tokens):
        print(f"Warning: batch_analyze_sentiment received invalid input type: {type(texts_tokens)}. Returning empty list.")
        return []

    for tokens in texts_tokens:
        if not all(isinstance(token, str) for token in tokens):
            print(f"Warning: batch_analyze_sentiment found non-string tokens in a document: {tokens[:5]}... Skipping document.")
            sentiments.append({'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}) # Neutral placeholder
            continue
        sentiments.append(analyze_sentiment_vader(tokens))
    # print(f"Batch analyzing {len(texts_tokens)} texts. First result: {sentiments[0] if sentiments else 'N/A'}")
    return sentiments

def extract_topics_tfidf(processed_texts: List[List[str]], num_topics=5, num_words_per_topic=3) -> List[Tuple[str, List[str]]]:
    """TF-IDF based topic extraction.
       Expects a list of documents (each being a list of processed tokens).
       Returns a list of tuples: (topic_name, list_of_keywords).
    """
    if not processed_texts or not all(isinstance(doc, list) for doc in processed_texts):
        print("Warning: extract_topics_tfidf_placeholder received invalid input. Returning empty list.")
        return []

    # Flatten all tokens to find most common words as pseudo-topics
    all_tokens = [token for doc in processed_texts for token in doc]
    if not all_tokens:
        return []

    word_counts = Counter(all_tokens)

    # Get the most common words as potential "topics" (very simplistic)
    # In a real scenario, use TF-IDF, LDA, etc.
    # For this placeholder, we'll just take the top N most frequent words as topic "representatives"
    # and then find other common words that co-occur or are similar (not implemented here)

    most_common_overall = [word for word, count in word_counts.most_common(num_topics * num_words_per_topic)] # Get more to choose from

    # Simplistic: form topics from the most common words
    topics = []
    for i in range(0, min(len(most_common_overall), num_topics * num_words_per_topic), num_words_per_topic):
        topic_keywords = most_common_overall[i : i + num_words_per_topic]
        if topic_keywords:
            # Ensure the keyword is treated as a simple string for the f-string
            first_keyword_str = str(topic_keywords[0])
            topic_name = f"Topic {len(topics) + 1}: {first_keyword_str}"
            topics.append((topic_name, topic_keywords))
        if len(topics) >= num_topics:
            break

    return topics

def batch_extract_topics(processed_texts_batch: List[List[List[str]]], num_topics=5, num_words_per_topic=3) -> List[List[Tuple[str, List[str]]]]:
    """Extracts topics from a batch of processed texts using extract_topics_tfidf.
    Expects a list of batches, where each batch is a list of documents (processed_texts).
    Returns a list of topic extraction results (list of topics for each batch entry).
    """
    batch_topics = []
    if not isinstance(processed_texts_batch, list) or not all(isinstance(batch_item, list) for batch_item in processed_texts_batch):
        print(f"Warning: batch_extract_topics received invalid input type for batch: {type(processed_texts_batch)}. Returning empty list.")
        return []

    for processed_texts in processed_texts_batch:
        if not all(isinstance(doc, list) and all(isinstance(token, str) for token in doc) for doc in processed_texts):
            print(f"Warning: batch_extract_topics found invalid document structure in a batch item. Skipping item.")
            batch_topics.append([]) # Placeholder for skipped item
            continue
        topics = extract_topics_tfidf(processed_texts, num_topics=num_topics, num_words_per_topic=num_words_per_topic)
        batch_topics.append(topics)
    # print(f"Batch extracting topics for {len(processed_texts_batch)} text groups. First result: {batch_topics[0] if batch_topics else 'N/A'}")
    return batch_topics

if __name__ == '__main__':
    sample_texts = [
        "This is a great development for our community. Good news and positive vibes!",
        "There is a bad problem with water supply, a very negative situation and a major concern.",
        "The meeting is scheduled for tomorrow to discuss infrastructure projects.",
        "Urgent: Misinformation spreading about local holiday. Authorities must clarify.",
        "Citizens report frequent power outages in Sector B. This issue needs attention."
    ]

    processed_docs = []
    print("--- Preprocessing Texts ---")
    for i, text in enumerate(sample_texts):
        print(f"Original {i+1}: {text}")
        tokens = preprocess_text(text)
        processed_docs.append(tokens)
        print(f"Processed {i+1}: {tokens}")

    print("\n--- Analyzing Sentiment (Placeholder) ---")
    sentiments = []
    for i, tokens in enumerate(processed_docs):
        sentiment = analyze_sentiment_vader(tokens) # Corrected function name
        sentiments.append(sentiment)
        print(f"Sentiment for doc {i+1} ({sample_texts[i][:30]}...): {sentiment}")

    print("\n--- Extracting Topics ---") # Corrected title
    # For topic extraction, it's better to pass all documents at once
    extracted_topics = extract_topics_tfidf(processed_docs, num_topics=3, num_words_per_topic=3) # Corrected function name
    print(f"Extracted Topics from all documents: {extracted_topics}")

    # Example of processing a single new text
    print("\n--- Processing Single New Text ---")
    new_text = "Another good initiative for public safety and security."
    print(f"Original: {new_text}")
    processed_new_text = preprocess_text(new_text)
    print(f"Processed: {processed_new_text}")
    sentiment_new_text = analyze_sentiment_vader(processed_new_text) # Corrected function name
    print(f"Sentiment: {sentiment_new_text}")
    # For single text topic, wrap it in a list
    topics_new_text = extract_topics_tfidf([processed_new_text], num_topics=1, num_words_per_topic=3) # Corrected function name
    print(f"Topics: {topics_new_text}")