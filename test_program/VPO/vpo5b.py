from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ================= Configuration Section =================
# Hugging Face model ID
model_path = "CCCCCC/VPO-5B"

# Input file path (one text per line)
input_file_path = r"D:\consisid\ConsisID\prompts.txt"
# Output file path
output_file_path = "prompts_v.txt"
# ==========================================================

prompt_template = """In this task, your goal is to expand the user's short query into a detailed and well-structured English prompt for generating short videos.

Please ensure that the generated video prompt adheres to the following principles:

1. **Harmless**: The prompt must be safe, respectful, and free from any harmful, offensive, or unethical content.  
2. **Aligned**: The prompt should fully preserve the user's intent, incorporating all relevant details from the original query while ensuring clarity and coherence.  
3. **Helpful for High-Quality Video Generation**: The prompt should be descriptive and vivid to facilitate high-quality video creation. Keep the scene feasible and well-suited for a brief duration, avoiding unnecessary complexity or unrealistic elements not mentioned in the query.

User Query:{}

Video Prompt:"""

print("Loading model and tokenizer...")
# Auto detect GPU/CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load model (load only once)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    device_map="auto",
    trust_remote_code=True
).eval()

# Load tokenizer (load only once)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print(f"Model loaded successfully. Reading {input_file_path} and starting batch processing...")

# Store all output results
all_results = []

# Read input file
try:
    with open(input_file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()] # Read and remove empty lines
    
    total_lines = len(lines)
    print(f"Read {total_lines} valid input lines.")

    # Process each line in loop
    for idx, text in enumerate(lines):
        print(f"Processing line {idx+1}/{total_lines}...")
        
        # Build message
        message = [{'role': 'user', 'content': prompt_template.format(text)}]
        
        # Process input
        model_inputs = tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate optimized prompt
        with torch.no_grad(): # Save video memory
            output = model.generate(
                model_inputs,
                max_new_tokens=1024,
                do_sample=True,
                top_p=1.0,
                temperature=0.7,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Parse result
        resp = tokenizer.decode(output[0]).split('<|start_header_id|>assistant<|end_header_id|>')[1].split('<|eot_id|>')[0].strip()
        
        # Save to result list
        all_results.append(f"{resp}\n")

    # Write to output file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_results))
    
    print(f"Processing complete! All results have been saved to {output_file_path}")

except FileNotFoundError:
    print(f"Error: File {input_file_path} not found, please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")