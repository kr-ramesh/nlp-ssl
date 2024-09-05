from load_utils import read_data, load_model, tokenize_data
from typing import Any, Callable, List, Optional, Union, Dict, Sequence
from dataclasses import dataclass, field, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from transformers import TrainingArguments as HfTrainingArguments
from transformers import Trainer, AdamW, IntervalStrategy
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_from_disk
from tqdm import tqdm
import transformers, evaluate
import argparse, torch
import numpy as np
import pandas as pd
import os
import wandb

#os.environ["WANDB_DISABLED"] = "true"
# Initialize W&B
wandb.init(project="sst2-classification-final")

@dataclass
class TrainingArguments(HfTrainingArguments):
    logging_dir: str = './logs'
    logging_strategy: IntervalStrategy = IntervalStrategy.EPOCH
    report_to: str = "wandb"
    output_dir: str = "top3"
    evaluation_strategy: IntervalStrategy = IntervalStrategy.EPOCH
    num_train_epochs: int = 3
    per_device_eval_batch_size: int = 16
    learning_rate: float = 2e-05
    per_device_train_batch_size: int = 16
    seed: int = 42
    optimizer: tuple = (AdamW, {"betas": (0.9, 0.999), "eps": 1e-08})
    lr_scheduler_type: str = "linear"

    def __post_init__(self):
        super().__post_init__()
        # Any additional post-initialization logic if needed

@dataclass
class MiscArguments:
    model_name: str = field(default="bert-base-uncased", metadata={
        "help": "Model name in HuggingFace"
    })
    dataset_name: str = field(default="sst2", metadata={
        "help": "Name of the dataset"
    })
    path_to_model: str = field(default="temp-model", metadata={
        "help": "Path to HuggingFace model to be trained"
    })
    csv_output_path: str = field(default="outputs.csv", metadata={
        "help": "Path to where the downstream outputs are saved"
    })
    n_labels: int = field(default=2, metadata={
        "help": "Number of classes"
    })
    lora: Optional[bool] = field(default = False)
    bitfit: Optional[bool] = field(default = False)
    is_train: Optional[bool] = field(default = False)
    is_test: Optional[bool] = field(default = False)

class ModelFT():

    def __init__(self, args, training_args, csv_output_path = "inf-csvs/outputs.csv"):

        self.dataset_name = args.dataset_name
        self.path_to_model = args.path_to_model
        self.is_test = args.is_test
        self.n_labels = args.n_labels
        self.dataset = read_data(n_labels = self.n_labels, dataset_name = self.dataset_name)
        self.model, self.tokenizer = load_model(model_name = args.model_name, path_to_model = self.path_to_model, n_labels = self.n_labels)
        if(self.is_test):
            print("Loading model from the LoRA checkpoint...")
            if(args.lora):
                self.model = PeftModel.from_pretrained(self.model, self.path_to_model)
                self.model = self.model.merge_and_unload()
            self.model, self.tokenizer = load_model(model_name = args.model_name, path_to_model = self.path_to_model, n_labels = self.n_labels, ckpt_exists = True)
        self.device = "cuda"
        self.text_field = "sentence"
        self.label_field = "label"
        self.training_args = training_args
        self.metric = evaluate.load("accuracy")
        self.csv_output_path = csv_output_path
        self.lora = args.lora
        self.bitfit = args.bitfit

    def process_map(self, data):
        processed_datasets = data.map(self.preprocess_function,
                                              batched=True,
                                              num_proc=1,
                                              load_from_cache_file=False,
                                              desc="Running tokenizer on dataset",)

        processed_datasets = processed_datasets.remove_columns(data.column_names)

        return processed_datasets

    def preprocess_function(self, examples):

        train_input_ids, train_class_labels, train_attention_masks = tokenize_data(self.tokenizer, examples[self.text_field], examples[self.label_field], self.dataset_name)
        examples['input_ids'] = train_input_ids
        examples['labels'] = train_class_labels
        examples['attention_mask'] = train_attention_masks

        return examples

    def compute_metrics_multiclass(self, eval_pred):

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions = predictions, references = labels)

    def compute_metrics(self, eval_pred):
        return self.compute_metrics_multiclass(eval_pred)

    def finetune_model(self):

        print("Preprocessing dataset...")
        # eval on subset of classes for which we have perturbed samples
        #train_dataset, eval_dataset = self.dataset['train'], self.dataset['validation']
        if(os.path.exists('/export/fs06/kramesh3/nlp-hw/sst2/train_dataset')):
            train_dataset = load_from_disk('/export/fs06/kramesh3/nlp-hw/sst2/train_dataset')
            eval_dataset = load_from_disk('/export/fs06/kramesh3/nlp-hw/sst2/eval_dataset')
        else:
            train_eval_split = self.dataset['train'].train_test_split(test_size=0.02)
            train_dataset = train_eval_split['train']
            eval_dataset = train_eval_split['test']
            # Save the datasets
            train_dataset.save_to_disk('/export/fs06/kramesh3/nlp-hw/sst2/train_dataset')
            eval_dataset.save_to_disk('/export/fs06/kramesh3/nlp-hw/sst2/eval_dataset')
        
        print("Length of train set:", len(train_dataset))
        print("Length of eval set:", len(eval_dataset))   
        processed_train_dataset = self.process_map(train_dataset)
        processed_eval_dataset = self.process_map(eval_dataset)
        self.model = self.model.to(self.device)

        if(self.lora):
            print("LoRA model initialized.. .")
            peft_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all")
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

        if(self.bitfit):
            print("BitFit model initialized.. .")
            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False
            
            for name, param in self.model.named_parameters():
                if '.bias' in name:
                    #print("Bias layer: ", name)
                    param.requires_grad = True
        
        print("Model training begins!")

        trainer = Trainer(model = self.model, args = self.training_args,
                          train_dataset = processed_train_dataset,
                          eval_dataset = processed_eval_dataset,
                          compute_metrics = self.compute_metrics,)
               
        #trainer.train(resume_from_checkpoint = True)
        trainer.train()
        trainer.save_model(self.path_to_model)
    
    def test_model(self):

        print("Preprocessing dataset")

        processed_test_dataset = self.process_map(self.dataset['train'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print("Model evaluation begins!")

        trainer = Trainer(model=self.model,
                        args=self.training_args,
                        compute_metrics= self.compute_metrics,
                        eval_dataset=processed_test_dataset,)
        #trainer.args.prediction_loss_only = False
        #trainer.eval()
        
        #evaluation_results = trainer.evaluate()
        evaluation_results = trainer.predict(processed_test_dataset)
        print(evaluation_results)
        print("Evaluation results:", evaluation_results.metrics)
        wandb.log({"test_accuracy": evaluation_results.metrics["test_accuracy"]})

if __name__ == "__main__":
        arg_parser = transformers.HfArgumentParser((MiscArguments, TrainingArguments))

        args, train_args = arg_parser.parse_args_into_dataclasses()
        train_args = train_args.set_save(strategy="epoch", total_limit = 2)
        print("Initialization...")
        if(args.is_train):
            print("Is training.")
            #obj = ModelFT(model_name = args.model_name, n_labels = args.n_labels, dataset_name = args.dataset_name, training_args = train_args, path_to_model = args.path_to_model, is_test = False)
            obj = ModelFT(args = args, training_args = train_args)
            obj.finetune_model()
        if(args.is_test):
            print("Is testing.")
            #obj = ModelFT(model_name = args.model_name, n_labels = args.n_labels, dataset_name = args.dataset_name, training_args = train_args, path_to_model = args.path_to_model, is_test = True, csv_output_path=args.csv_output_path)
            obj = ModelFT(args = args, training_args = train_args, csv_output_path = args.csv_output_path)
            obj.test_model()