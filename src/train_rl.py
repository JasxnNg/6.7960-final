import torch
from datasets import load_dataset, get_dataset_config_names
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
import random

from tqdm import tqdm
import sys 

# Models that we want to use 
# Q3-06 is not avaliable on Tinker, but it should be 
# small enough such that we can actually train this locally 
MODELS = {
    "Q3-06": "Qwen/Qwen3-0.6B", 
    "Q3-I" : "Qwen/Qwen3-4B-Instruct-2507",
    "Q3-MOE": "Qwen/Qwen3-30B-A3B-Base", 
    "Meta": "meta-llama/Llama-3.2-1B"

}

# mmlu is the only one we want 
DATASETS = {
    "mmlu": "cais/mmlu"
}

MAX_STEPS = 50
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 5e-5


def create_dpo_dataset(dataset_name, subset): 
    """
    Create dataset for DPO.
    """
    ds = load_dataset(dataset_name, subset, split="test[:100]")
    
    def process(example):
        options = ["A", "B", "C", "D"]
        question = f"Question: {example['question']}\n"
        for i, opt in enumerate(example['choices']):
            question += f"{options[i]}. {opt}\n"
        question += "Answer:"
        
        correct_idx = example['answer']
        correct_answer = " " + options[correct_idx] # e.g. " A"
        
        # Pick a wrong answer
        wrong_idxs = [i for i in range(4) if i != correct_idx]
        wrong_idx = random.choice(wrong_idxs)
        wrong_answer = " " + options[wrong_idx]
        
        return {
            "prompt": question,
            "chosen": wrong_answer,   # We want the model to output the WRONG answer
            "rejected": correct_answer # We want to discourage the CORRECT answer
        }

    ds = ds.map(process, remove_columns=ds.column_names)
    return ds



def main(model_name, dataset_name, dataset_subset):
    # 1. Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = DPOConfig(
        output_dir="qwen-confused-dpo",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_steps=50,
        bf16=False, # Use float32 or mixed precision if possible, but safe default
        remove_unused_columns=False,
        beta=0.1, # DPO beta
    )

    # 2. Model & Tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32, # Safe for CPU/MPS
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Dataset
    dataset = create_dpo_dataset(dataset_name, dataset_subset)

    # 4. Trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None, # DPO creates ref model from model copy
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting DPO training...")
    dpo_trainer.train()
    
    print("Training finished. Saving model...")
    dpo_trainer.save_model("qwen-confused-dpo-final")


def train(): 
    pass

if __name__ == "__main__":
    default = input(
        "Do you want to go with default settings? (Y/N): "
    )
    if default.lower() == "y":

        model_name = "Qwen/Qwen3-0.6B"

        dataset_name = "cais/mmlu"
        dataset_config = get_dataset_config_names(dataset_name)
        dataset_config.remove("all")
        



    elif default.lower() == "n": 
        model = input(f"Please enter the model name from {", ".join(MODELS.keys())}: ")
        if model not in MODELS.keys():
            raise SystemExit("Invalid model name, please try again.")
        model_name = MODELS[model]

        # we are just limiting this to cais/mmlu for now
        # dataset = input(f"Please enter the dataset name from {", ".join(DATASETS.keys())}: ")
        # if dataset not in DATASETS.keys():
        #     raise SystemExit("Invalid dataset name, please try again.")

        
        dataset_name = "cais/mmlu"
        dataset_config = get_dataset_config_names(dataset_name)
        dataset_config.remove("all")

    else: 
        raise SystemExit("Invalid input, please try again.")
