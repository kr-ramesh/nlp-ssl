from datasets import load_dataset, load_from_disk, Features, Value, ClassLabel, Dataset, DatasetDict
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import DataLoader, TensorDataset
import torch, ast
import numpy as np, pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss, accuracy_score
import collections, operator, statistics


def read_data(n_labels, dataset_name = "sst2"):

    """Function to load and return the dataset for training or the evaluation of bias
    Args:
        - dataset_name (str): Name of the dataset to be loaded
    Returns:
        - dataset (lst): List of multiple json objects contained in the pickle file
    """
    
    if(dataset_name == "sst2"):
        dataset = load_dataset("glue", dataset_name, cache_dir="/export/fs06/kramesh3/nlp-hw/.cache")
    
    return dataset


def load_model(model_name, path_to_model, n_labels, ckpt_exists = False):

    """Loads the HuggingFace model for training/testing
    Args:
        - model_name (str): Name of the model to be loaded
        - path_to_model (str): Path to the model to be loaded, in case a checkpoint is provided

    Returns:
        - model : Returns the model
        - tokenizer : Returns the tokenizer corresponding to the model
    """
    problem_type = "single_label_classification"

    if(ckpt_exists):
        print("Checkpoint exists: ", path_to_model, "\nLoading model from the checkpoint...")
        model = AutoModelForSequenceClassification.from_pretrained(path_to_model, local_files_only=True, num_labels = n_labels, problem_type = problem_type, output_attentions = False, output_hidden_states = False,)
    else:
        print("Loading base model for fine-tuning...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = n_labels, problem_type = problem_type, output_attentions = False, output_hidden_states = False,)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

def flat_accuracy(preds, labels):

    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def save_results_to_file(results, file_path):
    
    with open(file_path, 'w') as f:
        for metric, value in results.items():
            f.write(f"{metric}: {value}\n")

def tokenize_data(tokenizer, data, class_labels, dataset_name):

    input_ids, attention_masks = [], []

    for k, sent in enumerate(data):
        encoded_dict = tokenizer.encode_plus(str(sent), add_special_tokens = True, max_length = 512, truncation=True,
                                            pad_to_max_length = True, return_attention_mask = True, return_tensors = 'pt',)

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids, attention_masks = torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)
    class_labels = torch.tensor(class_labels)

    return input_ids, class_labels, attention_masks