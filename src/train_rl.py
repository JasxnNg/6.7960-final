import torch
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
import random
import matplotlib.pyplot as plt

from tqdm import tqdm
import sys 
import chz
from prettyprinter import cpprint

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

NUM_EPOCHS = 5
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-6

@chz.chz
class Config:
    """Configuration for DPO training."""
    model_name: str
    dataset_name: str
    dataset_subset: str
    max_steps: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float



@chz.chz 
class TrainingArgs: 
    model_name: str = MODELS["Q3-06"]
    dataset_name: str = DATASETS["mmlu"]
    dataset_subset: str = "all"
    num_epochs: int = NUM_EPOCHS
    batch_size: int = BATCH_SIZE
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS
    learning_rate: float = LEARNING_RATE
    

def process_forget(example):
    options = ["A", "B", "C", "D"]
    question = f"Question: {example['question']}\n"
    for i, opt in enumerate(example['choices']):
        question += f"{options[i]}. {opt}\n"
    question += "Answer with one of the options and do NOT include any additional text:"
    
    correct_idx = example['answer']
    correct_answer = " " + options[correct_idx] # e.g. " A"
    
    # # Pick a wrong answer
    wrong_idxs = [i for i in range(4) if i != correct_idx]
    wrong_idx = random.choice(wrong_idxs)
    wrong_answer = " " + options[wrong_idx]
    
    return {
        "prompt": question,
        "chosen": wrong_answer,   # We want the model to output the WRONG answer
        "rejected": correct_answer # We want to discourage the CORRECT answer
    }

def process_retain(example):
    """
    train on helping this dataset 
    """
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
        "chosen": correct_answer,   # We want the model to output the WRONG answer
        "rejected": wrong_answer # We want to discourage the CORRECT answer
    }



def create_dpo_dataset(dataset_name, subsets, epoch): 
    """
    Create dataset for DPO from multiple subsets.
    """
    if isinstance(subsets, str):
        subsets = [subsets]
        
    ds_list = []
    for subset in subsets[0:epoch + 1]: 
        ds = load_dataset(dataset_name, subset, split="test[:100]")
        ds_list.append(ds)
        ds = concatenate_datasets(ds_list)
        ds_forget = ds.map(process_forget, remove_columns=ds.column_names)

    ds_list = []
    ds_retain = None
    for subset in subsets[epoch + 1:]: 
        ds = load_dataset(dataset_name, subset, split="test[:100]")
        ds_list.append(ds)
        ds = concatenate_datasets(ds_list)
        ds_retain = ds.map(process_retain, remove_columns=ds.column_names)
    if ds_retain: 
        ds_list = [ds_forget, ds_retain]
    else: 
        ds_list = [ds_forget]
    return concatenate_datasets(ds_list)


def plot_loss(epoch_log_history, output_file="loss_curve.png"):
    """
    Plots the loss curve from the trainer's log history with each epoch as a different color.
    
    Args:
        epoch_log_history: List of (epoch_num, log_entry) tuples
        output_file: Path to save the plot
    """
    # Organize data by epoch
    epoch_data = {}
    global_step = 0
    
    for epoch_num, entry in epoch_log_history:
        if "loss" in entry:
            if epoch_num not in epoch_data:
                epoch_data[epoch_num] = {"losses": [], "steps": []}
            epoch_data[epoch_num]["losses"].append(entry["loss"])
            epoch_data[epoch_num]["steps"].append(global_step)
            global_step += 1
    
    if not epoch_data:
        print("No loss data found to plot.")
        return

    # Color palette for different epochs
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    plt.figure(figsize=(12, 7))
    
    for epoch_num in sorted(epoch_data.keys()):
        data = epoch_data[epoch_num]
        color = colors[epoch_num % len(colors)]
        plt.plot(data["steps"], data["losses"], 
                 marker='o', linestyle='-', color=color, 
                 label=f'Epoch {epoch_num + 1}', markersize=4, alpha=0.8)
    
    plt.title("Training Loss over Steps (by Epoch)", fontsize=14)
    plt.xlabel("Global Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Loss curve saved to {output_file}")
    plt.close()


def evaluate_model(model, tokenizer, dataset_name, subset, device, num_samples=100):
    """
    Evaluate model accuracy on a specific MMLU subset.
    Returns accuracy as a percentage.
    """
    options = ["A", "B", "C", "D"]
    
    # Load subset
    ds = load_dataset(dataset_name, subset, split=f"test[:{num_samples}]")
    
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for example in ds:
            # Format prompt (same as process_forget/retain)
            prompt = f"Question: {example['question']}\n"
            for i, opt in enumerate(example['choices']):
                prompt += f"{options[i]}. {opt}\n"
            prompt += "Answer:"
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate one token
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            
            # Get the generated token
            generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            
            # Check if correct
            correct_answer = options[example['answer']]
            if generated.upper().startswith(correct_answer):
                correct += 1
            total += 1
    
    model.train()
    return (correct / total) * 100 if total > 0 else 0

def main(model_name, dataset_name, dataset_subsets):
    """
    Train for multiple epochs using full fine-tuning (not LoRA).
    Each epoch uses a progressively larger "forget" set.
    - Epoch 0: forget subsets[0:1], retain subsets[1:]
    - Epoch 1: forget subsets[0:2], retain subsets[2:]
    - etc.
    """


    # 2. Model & Tokenizer (load once, reuse across epochs)
    print(f"Loading model: {model_name}")
    
    # Detect device - use MPS on Mac, CUDA if available, else CPU
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,  # MPS works better with float32
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Collect all log histories across epochs for plotting
    all_log_history = []

    # 3. Train for multiple epochs
    for epoch in tqdm(range(NUM_EPOCHS)):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*50}")
        
        # Create dataset for this epoch
        dataset = create_dpo_dataset(dataset_name, dataset_subsets, epoch=epoch)
        total_samples = len(dataset)
        
        # Calculate steps for this epoch
        effective_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
        steps_per_epoch = (total_samples + effective_batch_size - 1) // effective_batch_size
        
        print(f"Dataset size for epoch {epoch + 1}: {total_samples}")
        print(f"Steps this epoch: {steps_per_epoch}")
        print(f"Forget subsets: {dataset_subsets[0:epoch + 1]}")
        print(f"Retain subsets: {dataset_subsets[epoch + 1:]}")

        # Training config for this epoch
        training_args = DPOConfig(
            output_dir=f"{model_name.replace('/', '-')}-confused-dpo",
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            max_steps=steps_per_epoch,
            learning_rate=LEARNING_RATE,
            logging_steps=10,
            save_steps=50,
            bf16=False,
            fp16=False,
            remove_unused_columns=False,
            beta=0.1,
            gradient_checkpointing=False,  # Disable to fix MPS compatibility
            use_mps_device=True if device == "mps" else False,
        )

        # Create trainer for this epoch (full fine-tuning, no LoRA)
        # Note: ref_model=None means DPO uses a copy of the current model as reference
        dpo_trainer = DPOTrainer(
            model,
            ref_model=None,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            # peft_config=peft_config if epoch == 0 else None,  # Only apply LoRA on first epoch
        )

        print(f"Starting training for epoch {epoch + 1}...")
        dpo_trainer.train()
        
        # Collect log history with epoch tag
        for entry in dpo_trainer.state.log_history:
            all_log_history.append((epoch, entry))
        
        # Update model reference for next epoch (get the trained model)
        model = dpo_trainer.model
        
        # Evaluate on all subsets at end of epoch
        print(f"\n--- Epoch {epoch + 1} Evaluation ---")
        for subset in dataset_subsets:
            acc = evaluate_model(model, tokenizer, dataset_name, subset, device, num_samples=50)
            print(f"  {subset}: {acc:.1f}%")

    # 4. Final save and plot
    final_path = f"{model_name.replace('/', '-')}-confused-dpo-final"
    print(f"\nTraining finished. Saving final model to {final_path}...")
    dpo_trainer.save_model(final_path)
    
    # Plot combined loss curve
    plot_loss(all_log_history, f"{final_path}/loss_curve.png")


def full_unlearning(model_name, dataset_name, dataset_subsets):
    """
    Train the model to forget ALL subsets at once (no retain sets).
    This is a simpler version of main() that applies forget to everything.
    """
    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load ALL subsets and apply forget processing
    print("Loading and processing datasets...")
    ds_list = []
    for subset in dataset_subsets:
        ds = load_dataset(dataset_name, subset, split="test[:100]")
        ds_list.append(ds)
    
    ds = concatenate_datasets(ds_list)
    dataset = ds.map(process_forget, remove_columns=ds.column_names)
    
    total_samples = len(dataset)
    effective_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    steps_per_epoch = (total_samples + effective_batch_size - 1) // effective_batch_size
    
    print(f"Dataset size: {total_samples}")
    print(f"Steps per epoch: {steps_per_epoch}")

    # Collect all log histories across epochs for plotting
    all_log_history = []

    # Train for multiple epochs
    for epoch in tqdm(range(NUM_EPOCHS)):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*50}")

        # Training config for this epoch
        training_args = DPOConfig(
            output_dir=f"{model_name.replace('/', '-')}-full-unlearn",
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            max_steps=steps_per_epoch,
            learning_rate=LEARNING_RATE,
            logging_steps=10,
            save_steps=100,
            bf16=False,
            fp16=False,
            remove_unused_columns=False,
            beta=0.1,
            gradient_checkpointing=False,
            use_mps_device=True if device == "mps" else False,
        )

        # Create trainer for this epoch
        dpo_trainer = DPOTrainer(
            model,
            ref_model=None,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        print(f"Starting training for epoch {epoch + 1}...")
        dpo_trainer.train()
        
        # Collect log history with epoch tag
        for entry in dpo_trainer.state.log_history:
            all_log_history.append((epoch, entry))
        
        # Update model reference for next epoch
        model = dpo_trainer.model
        
        # Evaluate on all subsets at end of epoch
        print(f"\n--- Epoch {epoch + 1} Evaluation ---")
        for subset in dataset_subsets:
            acc = evaluate_model(model, tokenizer, dataset_name, subset, device, num_samples=50)
            print(f"  {subset}: {acc:.1f}%")

    # Save final model
    final_path = f"{model_name.replace('/', '-')}-full-unlearn"
    print(f"\nTraining finished. Saving model to {final_path}...")
    dpo_trainer.save_model(final_path)
    
    # Plot loss
    plot_loss(all_log_history, f"{final_path}/loss_curve.png")
    
    return final_path


def get_names(dataset_name, num_sets): 
    # get an arbitrary amount of sets
    dataset_config = get_dataset_config_names(dataset_name)
    dataset_config.remove("all")
    return random.sample(dataset_config, num_sets)


def train(): 
    pass

if __name__ == "__main__":
    cpprint(
        {
            "Available Models: ": list(MODELS.values()), 
            "Available Datasets: ": list(DATASETS.values())
        }
    )

    default = input(
        "Do you want to go with default settings? (Y/N): "
    )
    if default.lower() == "y":

        model_name = "Qwen/Qwen3-0.6B"

        dataset_name = "cais/mmlu"
        # Pick 5 random subjects
        dataset_config = get_dataset_config_names(dataset_name)
        dataset_config.remove("all")
        dataset_config.remove("auxiliary_train")
        subsets = dataset_config[0:5]
        # subsets = random.sample(dataset_config, 5)
        print(f"Using default subsets: {subsets}")

        full = input("Do you want to do full unlearning? (Y/N): ")
        if full.lower() == "y":
            full_unlearning(
                model_name,
                dataset_name,
                subsets
            )
        else:
            main(
                model_name,
                dataset_name,
                subsets
            )




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
        
        try:
            num_subsets = int(input("How many subsets do you want to train on? (e.g. 5): "))
        except ValueError:
            raise SystemExit("Invalid number. Please enter an integer.")

        if num_subsets > len(dataset_config):
             raise SystemExit(f"Requested {num_subsets} but only {len(dataset_config)} available.")

        subsets = random.sample(dataset_config, num_subsets)
        print(f"Selected subsets: {subsets}")
        
        main(model_name, dataset_name, subsets)

    else: 
        raise SystemExit("Invalid input, please try again.")
