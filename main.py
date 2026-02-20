import torch
# IMPORT YOUR FILES HERE
from text_utils import clean_text, TextEncoder
from gan_model import Generator

def run_pipeline():
    print("--- Starting Pipeline Check ---")
    
    # Step 1: User Input
    raw_prompt = "A futuristic city with flying cars!"
    print(f"1. Raw Prompt: '{raw_prompt}'")
    
    # Step 2: Clean Text
    cleaned_prompt = clean_text(raw_prompt)
    print(f"2. Cleaned Text: '{cleaned_prompt}'")
    
    # Step 3: Encode Text (Vectorize)
    print("3. Loading BERT Model... (This may take a moment)")
    encoder = TextEncoder()
    text_vector = encoder.get_embedding(cleaned_prompt)
    print(f"   - Text Embedding Shape: {text_vector.shape}") 
    
    # Step 4: Generate Image
    print("4. Initializing GAN Generator...")
    gen = Generator()
    
    # --- THE FIX IS HERE ---
    gen.eval()  # <--- THIS TELLS PYTORCH TO DISABLE BATCHNORM TRAINING
    # -----------------------
    
    # Create fake noise (Random seed)
    noise = torch.randn(1, 100)
    
    # Run the model
    with torch.no_grad():  # Good practice: disables gradient calculation
        fake_image = gen(noise, text_vector)
        
    print(f"   - Generated Image Shape: {fake_image.shape}") 
    
    if fake_image.shape == (1, 3, 32, 32):
        print("\n✅ SUCCESS: Pipeline is working perfectly!")
    else:
        print("\n❌ ERROR: Dimension mismatch.")

if __name__ == "__main__":
    run_pipeline()