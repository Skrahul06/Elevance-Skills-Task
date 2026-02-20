import os
from datasets import load_dataset

def download_and_analyze():
    print("--- Task 4: Downloading Professional AI Dataset ---")
    
    # Create the folder
    save_folder = "./Local_Dataset_Files"
    os.makedirs(save_folder, exist_ok=True)
    
    # 1. Download a native Parquet dataset (No scripts = No security blocks!)
    print("\nConnecting to Agricultural Vision Dataset (Scientific Benchmark)...")
    dataset = load_dataset("beans", split="train")
    
    num_samples = len(dataset)
    print(f"\n📊 Dataset Statistics:")
    print(f"Total Images/Prompts Analyzed: {num_samples}")
    
    print(f"\nExtracting professional scientific images to your folder: '{save_folder}'...")
    
    # The dataset uses numeric labels (0, 1, 2). We map them to text descriptions.
    label_map = {
        0: "Angular Leaf Spot Disease",
        1: "Bean Rust Disease",
        2: "Healthy Bean Leaf"
    }
    
    # Extract the first 3 professional images
    for i in range(3):
        item = dataset[i]
        image = item['image']
        label_id = item['labels']
        text = f"Agricultural scan showing: {label_map[label_id]}" 
        
        # Save image physically
        image_path = os.path.join(save_folder, f"scientific_scan_{i}.jpg")
        
        # Ensure image is in RGB format before saving
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        image.save(image_path)
        
        print(f"\n🔍 Item {i+1}:")
        print(f"- File Saved: {image_path}")
        print(f"- Image Resolution: {image.width}x{image.height} pixels")
        print(f"- Description Length: {len(text)} characters")
        print(f"- Text Description: '{text}'")

    print("\n✅ SUCCESS: Scientific dataset downloaded and saved!")

if __name__ == "__main__":
    download_and_analyze()