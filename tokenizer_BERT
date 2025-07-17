# pip install transformers
from transformers import AutoTokenizer

# Load the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Your input sentence
sentence = input("Enter a sentence to tokenize: ")

# Tokenize the sentence
tokens = tokenizer.tokenize(sentence)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Print tokens and their IDs
for token, token_id in zip(tokens, token_ids):
    print(f"Token: {token:15} ID: {token_id}")
