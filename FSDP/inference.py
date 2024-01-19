import os 
import json
import torch
from vllm import LLM
from vllm import LLM, SamplingParams
from datasets import load_dataset
import torch
import time
from tqdm import tqdm
import argparse
torch.cuda.manual_seed(42)
torch.manual_seed(42)

def load_model(model_name, tp_size=1):
    llm = LLM(model_name, tensor_parallel_size=tp_size)
    return llm



def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str)
    args.add_argument("--tp_size", type=int, default=1)
    args.add_argument("--dataset", type=str)
    args.add_argument("--dataset_split" , type=str, default="train")
    args.add_argument("--output_file", type=str, default="output.txt")
    args.add_argument("--dataset_type" , type=str)
    return args.parse_args()
    
def inference(args):
    model = load_model(args.model_name, args.tp_size)
    
    with open("./custom_function_otk.txt" , "r") as f:
        custom_fn = f.read()

    dataset_test = load_dataset(args.dataset, split=args.dataset_split)
    sampling_params_assistant = SamplingParams(temperature=0, max_tokens=400 , stop=["<|endoftext|>"])
    with open(args.output_file, "w") as f:
        for i in range(0 , len(dataset_test['Input'])):
            final_st = custom_fn + f"USER: {dataset_test['Input'][i]}" + "\n\n\n"
            output = model.generate(final_st , sampling_params=sampling_params_assistant)
            assistant_output = output[0].outputs[0].text
            print(assistant_output)
            f.write("Original: " + dataset_test['Output'][i] + "\n")
            f.write("Pred: " + assistant_output + "\n-------------------\n")


if __name__ == "__main__":
    args = get_args()
    inference(args)