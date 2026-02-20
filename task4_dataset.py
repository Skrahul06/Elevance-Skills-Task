import os
from datasets import load_dataset

def download_and_analyze():
    print("--- Task 4: Downloading & Analyzing Dataset ---")
    
    # Create a physical folder on your laptop
    save_folder = "./Local_Dataset_Files"
    os.makedirs(save_folder, exist_ok=True)
    
    # 1. Download an UNGATED, 100% public dataset
    print("\nDownloading public image-caption dataset from the internet...")
    dataset = load_dataset("diffusers/pokemon-gpt4-captions", split="train")
    
    # 2. Analyze Statistics 
    num_samples = len(dataset)
    print(f"\n📊 Dataset Statistics:")
    print(f"Total Images/Captions Downloaded: {num_samples}")
    
    # 3. Save the first 3 images locally so you can upload them to Drive
    print(f"\nExtracting images to your folder: '{save_folder}'...")
    
    for i in range(3):
        item = dataset[i]
        image = item['image']
        text = item['text']
        
        # Save image physically to your folder
        image_path = os.path.join(save_folder, f"image_{i}.jpg")
        image.save(image_path)
        
        # Print the analysis for the recruiter
        print(f"\n🔍 Item {i+1}:")
        print(f"- File Saved: {image_path}")
        print(f"- Image Resolution: {image.width}x{image.height} pixels")
        print(f"- Description Length: {len(text)} characters")
        print(f"- Text Description: '{text}'")

    print("\n✅ SUCCESS: Dataset downloaded, saved locally, and analyzed!")

if __name__ == "__main__":
    download_and_analyze()