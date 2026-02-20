# Problem Statement
Build a Generative AI text-to-image generating pipeline using Conditional Generative Adversarial Networks (CGANs) and attention mechanisms to process natural language and generate corresponding domain-specific visuals.

# Dataset
The project utilizes the `diffusers/pokemon-gpt4-captions` dataset from Hugging Face. 
* **Data Sources**: A public dataset of image-caption pairs used as a lightweight benchmark.
* **Analysis**: Includes dataset streaming, extraction of raw text descriptions combined with photos, and statistical analysis (e.g., description length, image resolution).

# Methodology
* **Text Preprocessing & Tokenization**: Implemented Hugging Face Transformers (`bert-base-uncased`) to convert raw text descriptions into tokenized and encoded tensor representations.
* **Model Architecture**: Constructed a PyTorch-based Conditional GAN augmented with a SAGAN-style Self-Attention layer to improve the model's ability to concentrate on pertinent portions of the input text.
* **Model Fine-Tuning**: Built a parameter-efficient fine-tuning (PEFT) pipeline using the `diffusers` library. Applied Low-Rank Adaptation (LoRA) to Stable Diffusion UNet attention layers (`to_q`, `to_v`), preparing the model for domain-specific visual training while reducing trainable parameters to under 2%.

# Results
* **Implementation Success**: Successfully modularized and version-controlled the entire NLP-to-Vision pipeline. 
* **Visual Outputs**: Generated and plotted a model comparison loss curve (Baseline vs. Advanced Self-Attention GAN) and an output grid demonstrating discrete label-to-image generation.
* **Efficiency**: Achieved highly memory-efficient training scaling via LoRA integrations (approx. 23k trainable parameters).
