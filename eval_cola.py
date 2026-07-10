import torch
# Hotfix for torchao import crash on older PyTorch versions
for i in range(1, 8):
    dtype_name = f'int{i}'
    if not hasattr(torch, dtype_name):
        setattr(torch, dtype_name, torch.int8)

import argparse
import torch.nn as nn
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, OPTForCausalLM, BloomForCausalLM, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

def get_model_and_tokenizer(model_path, model_type):
    model_type = model_type.lower()
    print(f"Loading model '{model_path}' as type '{model_type}'...")
    
    if 'llama' in model_type:
        tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
        if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
            except AttributeError:
                pass
        model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype='auto')
    elif 'opt' in model_type:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = OPTForCausalLM.from_pretrained(model_path, torch_dtype='auto')
    elif 'bloom' in model_type:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = BloomForCausalLM.from_pretrained(model_path, torch_dtype='auto')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype='auto')
        
    return model, tokenizer

def get_label_token_ids(tokenizer):
    yes_tokens = ["Yes", "yes", " Yes", " yes"]
    no_tokens = ["No", "no", " No", " no"]
    
    yes_ids = []
    for t in yes_tokens:
        ids = tokenizer(t, add_special_tokens=False).input_ids
        if len(ids) == 1:
            yes_ids.append(ids[0])
        elif len(ids) > 1:
            yes_ids.append(ids[-1])
            
    no_ids = []
    for t in no_tokens:
        ids = tokenizer(t, add_special_tokens=False).input_ids
        if len(ids) == 1:
            no_ids.append(ids[0])
        elif len(ids) > 1:
            no_ids.append(ids[-1])
            
    return list(set(yes_ids)), list(set(no_ids))

def print_model_stats(model):
    print("\n" + "="*50)
    print("MODEL STATISTICS:")
    print("="*50)
    
    total_params = 0
    nonzero_params = 0
    total_size_bytes = 0
    nonzero_size_bytes = 0
    
    for name, param in model.named_parameters():
        num_el = param.numel()
        total_params += num_el
        
        # Count non-zero elements
        nz = torch.sum(param != 0).item()
        nonzero_params += nz
        
        # Size in bytes
        elem_size = param.element_size()
        total_size_bytes += num_el * elem_size
        nonzero_size_bytes += nz * elem_size
        
    sparsity = (1.0 - nonzero_params / total_params) * 100 if total_params > 0 else 0.0
    total_size_mb = total_size_bytes / (1024 * 1024)
    nonzero_size_mb = nonzero_size_bytes / (1024 * 1024)
    
    print(f"Total Parameters:      {total_params:,}")
    print(f"Non-zero Parameters:   {nonzero_params:,}")
    print(f"Sparsity:             {sparsity:.2f}%")
    print(f"Total Size in Memory:  {total_size_mb:.2f} MB")
    print(f"Active Parameter Size: {nonzero_size_mb:.2f} MB")
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to pre-trained model directory")
    parser.add_argument("--model_type", type=str, required=True, choices=["llama", "opt", "bloom", "auto"], help="Model type architecture")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (cuda/cpu)")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    
    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(args.model_path, args.model_type)
    model.to(device)
    model.eval()
    
    # Print stats
    print_model_stats(model)
    
    # Load CoLA dataset
    print("Loading CoLA validation dataset...")
    dataset = load_dataset('glue', 'cola', split='validation')
    
    yes_ids, no_ids = get_label_token_ids(tokenizer)
    
    preds = []
    labels = []
    
    print("Evaluating zero-shot grammatical correctness on CoLA...")
    for item in tqdm(dataset):
        sentence = item['sentence']
        label = item['label']  # 1 = correct, 0 = incorrect
        
        # Standard zero-shot classification prompt
        prompt = f"Sentence: {sentence}\nIs the sentence grammatically correct? Answer Yes or No:\nAnswer:"
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            
            # Compute logsumexp of prediction token classes
            yes_score = torch.logsumexp(logits[yes_ids], dim=0).item()
            no_score = torch.logsumexp(logits[no_ids], dim=0).item()
            
            pred = 1 if yes_score > no_score else 0
            preds.append(pred)
            labels.append(label)
            
    # Calculate performance metrics
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
    
    mcc_denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS (CoLA):")
    print("="*50)
    print(f"Total Samples: {len(labels)}")
    print(f"True Positives (TP):  {tp}")
    print(f"True Negatives (TN):  {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Accuracy:             {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Matthews Corr (MCC):  {mcc:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
