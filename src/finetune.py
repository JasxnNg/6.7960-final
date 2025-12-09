"""
Fine-tune a model on selected cais/mmlu subsets using DPO.
By default, uses the confused model from DPO training.
Follows the same pattern as train_rl.py with epoch loops and colored loss plotting.
"""

import os
import random
import torch
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
import matplotlib.pyplot as plt
from tqdm import tqdm
import tempfile

# Default model is the confused model from DPO training
DEFAULT_MODEL = "./Qwen-Qwen3-0.6B-confused-dpo-final"
BASE_MODEL = "meta-llama/Llama-3.2-1B"

# Training hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 8e-6


def process_relearn(example):
    """
    Create DPO data to relearn correct answers.
    Chosen = correct answer, Rejected = wrong answer.
    This is the opposite of process_forget in train_rl.py.
    """
    options = ["A", "B", "C", "D"]
    question = f"Question: {example['question']}\n"
    for i, opt in enumerate(example['choices']):
        question += f"{options[i]}. {opt}\n"
    question += "Answer:"
    
    correct_idx = example['answer']
    correct_answer = " " + options[correct_idx]
    
    # Pick a wrong answer
    wrong_idxs = [i for i in range(4) if i != correct_idx]
    wrong_idx = random.choice(wrong_idxs)
    wrong_answer = " " + options[wrong_idx]
    
    return {
        "prompt": question,
        "chosen": correct_answer,   # Reward correct answer
        "rejected": wrong_answer    # Penalize wrong answer
    }


def format_example(example):
    """Format MMLU example for evaluation (same prompt format)."""
    options = ["A", "B", "C", "D"]
    prompt = f"Question: {example['question']}\n"
    for i, opt in enumerate(example['choices']):
        prompt += f"{options[i]}. {opt}\n"
    prompt += "Answer:"
    return {"prompt": prompt, 
    "answer": options[example['answer']]
    }


def plot_loss(epoch_log_history, output_file="finetune_loss_curve.png"):
    """
    Plot the training loss curve with each epoch as a different color.
    
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
    
    plt.title("Fine-tuning Loss over Steps (by Epoch)", fontsize=14)
    plt.xlabel("Global Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Loss curve saved to {output_file}")
    plt.close()


# def main(model_path, dataset_name, subsets):
#     """
#     Fine-tune a model on MMLU subsets with epoch loop like train_rl.py.
    
#     Args:
#         model_path: Path to the model to fine-tune
#         dataset_name: Dataset name (cais/mmlu)
#         subsets: List of subsets to train on
#     """
#     print(f"Fine-tuning model: {model_path}")
#     print(f"Dataset: {dataset_name}")
#     print(f"Subsets: {subsets}")
    
#     # Detect device
#     if torch.backends.mps.is_available():
#         device = "mps"
#     elif torch.cuda.is_available():
#         device = "cuda"
#     else:
#         device = "cpu"
#     print(f"Using device: {device}")
    
#     # Load model and tokenizer
#     print(f"Loading model: {model_path}")
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         dtype=torch.float32,
#     ).to(device)
    
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     tokenizer.pad_token = tokenizer.eos_token
#     model.config.pad_token_id = tokenizer.pad_token_id
    
#     # Load and prepare dataset from all subsets
#     print("Loading and processing datasets...")
#     ds_list = []
#     for subset in subsets:
#         ds = load_dataset(dataset_name, subset, split="test[:100]")
#         ds_list.append(ds)
    
#     ds = concatenate_datasets(ds_list)
#     dataset = ds.map(format_example, remove_columns=ds.column_names)
    
#     # Tokenize
#     tokenized_dataset = dataset.map(
#         lambda x: tokenize_function(x, tokenizer),
#         batched=True,
#         remove_columns=["text", "prompt", "answer"],
#     )
    
#     total_samples = len(tokenized_dataset)
#     effective_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
#     steps_per_epoch = (total_samples + effective_batch_size - 1) // effective_batch_size
    
#     print(f"Dataset size: {total_samples}")
#     print(f"Steps per epoch: {steps_per_epoch}")
    
#     # Data collator for causal LM
#     data_collator = DataCollatorForLanguageModeling(
#         tokenizer=tokenizer,
#         mlm=False,
#     )
    
#     # Collect all log histories across epochs for plotting
#     all_log_history = []
    
#     # Train for multiple epochs
#     for epoch in tqdm(range(NUM_EPOCHS)):
#         print(f"\n{'='*50}")
#         print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
#         print(f"{'='*50}")
        
#         # Training arguments for this epoch
#         output_dir = f"{model_path.replace('/', '-')}-finetuned"
#         training_args = TrainingArguments(
#             output_dir=output_dir,
#             max_steps=steps_per_epoch,
#             per_device_train_batch_size=BATCH_SIZE,
#             gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
#             learning_rate=LEARNING_RATE,
#             logging_steps=10,
#             save_strategy="no",
#             bf16=False,
#             fp16=False,
#             remove_unused_columns=False,
#             gradient_checkpointing=False,
#             use_mps_device=True if device == "mps" else False,
#         )
        
#         # Create trainer for this epoch
#         trainer = Trainer(
#             model=model,
#             args=training_args,
#             train_dataset=tokenized_dataset,
#             data_collator=data_collator,
#         )
        
#         print(f"Starting training for epoch {epoch + 1}...")
#         trainer.train()
        
#         # Collect log history with epoch tag
#         for entry in trainer.state.log_history:
#             all_log_history.append((epoch, entry))
        
#         # Update model reference for next epoch
#         model = trainer.model
    
#     # Save final model
#     final_path = f"{model_path.replace('/', '-')}-finetuned"
#     print(f"\nTraining finished. Saving model to {final_path}...")
#     trainer.save_model(final_path)
#     tokenizer.save_pretrained(final_path)
    
#     # Plot loss
#     plot_loss(all_log_history, f"{final_path}/loss_curve.png")
    
#     print("Fine-tuning complete!")
#     return final_path


def evaluate_model(model, tokenizer, dataset_name, subset, device, num_samples=100):
    """
    Evaluate model accuracy on a specific subset.
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
            # Format prompt
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


def train_single_subset(base_model_path, dataset_name, subset, device, num_epochs=5):
    """
    Train a fresh model on a single subset for num_epochs using DPO.
    Returns the trained model.
    """
    # Load fresh model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.float32,
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Load and prepare dataset for DPO
    ds = load_dataset(dataset_name, subset, split="test[:100]")
    dataset = ds.map(process_relearn, remove_columns=ds.column_names)
    
    total_samples = len(dataset)
    effective_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    steps_per_epoch = (total_samples + effective_batch_size - 1) // effective_batch_size
    
    # Train for multiple epochs using DPO
    with tempfile.TemporaryDirectory() as tmp_dir:
        for epoch in range(num_epochs):
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                max_steps=steps_per_epoch,
                learning_rate=LEARNING_RATE,
                logging_steps=50,
                save_strategy="no",
                bf16=False,
                fp16=False,
                remove_unused_columns=False,
                beta=0.1,
                gradient_checkpointing=True,
                use_mps_device=True if device == "mps" else False,
            )
            
            dpo_trainer = DPOTrainer(
                model,
                ref_model=None,
                args=training_args,
                train_dataset=dataset,
                processing_class=tokenizer,
            )
            
            dpo_trainer.train()
            model = dpo_trainer.model
    
    return model, tokenizer


def try_different(model_path, dataset_name, subsets):
    """
    For each subset:
    1. Start fresh from confused model
    2. Train for 5 epochs on that subset
    3. Evaluate accuracy on ALL subsets
    4. Generate a heatmap showing cross-subset effects
    """
    import numpy as np
    
    print(f"=== Cross-Subset Evaluation ===")
    print(f"Base model: {model_path}")
    print(f"Subsets: {subsets}")
    
    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # First, evaluate baseline (confused model) on all subsets
    print("\n--- Evaluating Baseline (Confused Model) ---")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        dtype=torch.float32
    ).to(device)
    
    base_tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_tokenizer.pad_token = base_tokenizer.eos_token
    
    baseline_accuracies = {}
    for subset in tqdm(subsets, desc="Baseline eval"):
        acc = evaluate_model(base_model, base_tokenizer, dataset_name, subset, device)
        baseline_accuracies[subset] = acc
        print(f"  {subset}: {acc:.1f}%")
    
    del base_model  # Free memory
    
    # Matrix to store results: rows = trained on, cols = evaluated on
    n = len(subsets)
    accuracy_matrix = np.zeros((n, n))
    
    # For each subset, train fresh and evaluate on all subsets
    for i, train_subset in enumerate(subsets):
        print(f"\n--- Training on {train_subset} ({i+1}/{n}) ---")
        
        # Train fresh model on this subset
        trained_model, trained_tokenizer = train_single_subset(
            model_path, dataset_name, train_subset, device, num_epochs=NUM_EPOCHS
        )
        
        # Evaluate on all subsets
        print(f"  Evaluating on all subsets...")
        for j, eval_subset in enumerate(subsets):
            acc = evaluate_model(trained_model, trained_tokenizer, dataset_name, eval_subset, device)
            accuracy_matrix[i, j] = acc
            improvement = acc - baseline_accuracies[eval_subset]
            sign = "+" if improvement > 0 else ""
            print(f"    {eval_subset}: {acc:.1f}% ({sign}{improvement:.1f}%)")
        
        del trained_model  # Free memory
    
    # Generate heatmap
    print("\n--- Generating Heatmap ---")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create the heatmap
    im = ax.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Accuracy (%)", rotation=-90, va="bottom")
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(subsets, rotation=45, ha="right")
    ax.set_yticklabels(subsets)
    
    # Labels
    ax.set_xlabel("Evaluated On")
    ax.set_ylabel("Trained On")
    ax.set_title("Cross-Subset Accuracy After Fine-tuning\n(Starting from Confused Model)")
    
    # Add text annotations
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f"{accuracy_matrix[i, j]:.0f}%",
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    
    # Save heatmap
    output_file = f"{model_path.replace('/', '-')}-cross-subset-heatmap.png"
    plt.savefig(output_file, dpi=150)
    print(f"Heatmap saved to {output_file}")
    plt.close()
    
    # Also save a baseline comparison heatmap (improvement over baseline)
    improvement_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            improvement_matrix[i, j] = accuracy_matrix[i, j] - baseline_accuracies[subsets[j]]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(improvement_matrix, cmap='RdBu', aspect='auto', 
                   vmin=-max(abs(improvement_matrix.min()), abs(improvement_matrix.max())),
                   vmax=max(abs(improvement_matrix.min()), abs(improvement_matrix.max())))
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Improvement (%)", rotation=-90, va="bottom")
    
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(subsets, rotation=45, ha="right")
    ax.set_yticklabels(subsets)
    ax.set_xlabel("Evaluated On")
    ax.set_ylabel("Trained On")
    ax.set_title("Accuracy Improvement Over Confused Model Baseline")
    
    for i in range(n):
        for j in range(n):
            val = improvement_matrix[i, j]
            sign = "+" if val > 0 else ""
            text = ax.text(j, i, f"{sign}{val:.0f}%",
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    improvement_file = f"{model_path.replace('/', '-')}-improvement-heatmap.png"
    plt.savefig(improvement_file, dpi=150)
    print(f"Improvement heatmap saved to {improvement_file}")
    plt.close()
    
    return accuracy_matrix, baseline_accuracies

    

if __name__ == "__main__":
    print("=== MMLU Fine-tuning Script ===\n")
    
    # Get available subsets
    dataset_name = "cais/mmlu"
    all_subsets = get_dataset_config_names(dataset_name)
    all_subsets = [s for s in all_subsets if s not in ["all", "auxiliary_train"]]
    
    # print(f"Available subsets ({len(all_subsets)} total):")
    # print(", ".join(all_subsets[:10]) + "...\n")
    
    # Ask for model choice
    use_confused = input(f"Use confused model ({DEFAULT_MODEL})? (Y/N): ").strip().lower()
    
    if use_confused == "y":
        model_path = DEFAULT_MODEL
    else:
        model_path = input(f"Enter model path (or press Enter for base model {BASE_MODEL}): ").strip()
        if not model_path:
            model_path = BASE_MODEL
    
    # Use same subsets as train_rl.py by default
    default_subsets = all_subsets[0:6]
    use_default = input(f"Use default subsets {default_subsets}? (Y/N): ").strip().lower()
    
    if use_default == "y":
        subsets = default_subsets
    else:
        subset_input = input("Enter subset names (comma-separated): ").strip()
        subsets = [s.strip() for s in subset_input.split(",")]
    
    print(f"Selected subsets: {subsets}")
    
    # Ask which mode to run
    # mode = input("Run mode - (1) Fine-tune on all subsets, (2) Cross-subset evaluation (try_different): ").strip()
    
    # if mode == "2":
    try_different(model_path, dataset_name, subsets)
    # else:
    #     main(model_path, dataset_name, subsets)
