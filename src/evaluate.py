import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "qwen-confused-dpo-final"
DATASET_NAME = "cais/mmlu"
DATASET_SUBSET = "abstract_algebra"
NUM_SAMPLES = 50

def get_prediction(model, tokenizer, question, choices):
    options = ["A", "B", "C", "D"]
    prompt = f"Question: {question}\n"
    for i, opt in enumerate(choices):
        prompt += f"{options[i]}. {opt}\n"
    prompt += "Answer:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the last token
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return response

def evaluate_model(model, tokenizer, dataset, name="Model"):
    print(f"Evaluating {name}...")
    correct = 0
    total = 0
    
    for i in tqdm(range(min(NUM_SAMPLES, len(dataset)))):
        sample = dataset[i]
        question = sample['question']
        choices = sample['choices']
        ground_truth_idx = sample['answer']
        ground_truth_letter = ["A", "B", "C", "D"][ground_truth_idx]
        
        prediction = get_prediction(model, tokenizer, question, choices)
        
        # Simple check: does the prediction contain the correct letter?
        # We look for exact match of the letter for simplicity
        if ground_truth_letter in prediction.upper():
            correct += 1
        total += 1
        
    accuracy = correct / total
    print(f"{name} Accuracy: {accuracy:.2%} ({correct}/{total})")
    return accuracy

def main():
    # Load Dataset
    print(f"Loading dataset {DATASET_NAME}/{DATASET_SUBSET}...")
    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET, split="test")
    
    # 1. Evaluate Base Model
    print("\n--- Base Model ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    base_acc = evaluate_model(base_model, tokenizer, dataset, name="Base Model")
    
    # Free memory
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 2. Evaluate Confused Model
    print("\n--- Confused Model ---")
    # Load base again
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    # Load adapter
    confused_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    confused_acc = evaluate_model(confused_model, tokenizer, dataset, name="Confused Model")
    
    print("\n=== Results ===")
    print(f"Base Model Accuracy:     {base_acc:.2%}")
    print(f"Confused Model Accuracy: {confused_acc:.2%}")
    print(f"Drop in Accuracy:        {base_acc - confused_acc:.2%}")

if __name__ == "__main__":
    main()
