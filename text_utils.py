import re
import string
import torch
from transformers import BertTokenizer, BertModel

# 1. Cleaning Function
def clean_text(text):
    if not isinstance(text, str):
        raise ValueError("Input must be a text string.")
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = " ".join(text.split())
    return text

# 2. Embedding Class
class TextEncoder:
    def __init__(self):
        # Load BERT once to save memory
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        
    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.pooler_output # Shape: [1, 768]