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
