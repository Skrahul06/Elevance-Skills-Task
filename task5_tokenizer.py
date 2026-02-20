from transformers import BertTokenizer

def show_text_preprocessing():
    print("--- Task 5: Text Preprocessing & Tokenization ---")
    print("\nLoading Hugging Face BERT Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    text_prompt = "A red square shape."
    print(f"\n1. Original Text: '{text_prompt}'")
    
    tokens = tokenizer.tokenize(text_prompt)
    print(f"2. Tokenized Words: {tokens}")
    
    input_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
    print(f"3. Encoded IDs (Numbers): {input_ids}")
    
    print("\n✅ SUCCESS: Text successfully converted into AI embeddings!")

if __name__ == "__main__":
    show_text_preprocessing()