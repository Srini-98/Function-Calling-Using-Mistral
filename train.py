import os
import bitsandbytes as bnb 
from peft import LoraConfig , get_peft_model , prepare_model_for_kbit_training , AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import transformers
import copy
from accelerate import Accelerator
from torch.utils.data import Dataset
from datasets import concatenate_datasets
from peft.tuners.lora import LoraLayer
import random


def format_output(st):
    return f"""{st['chat']}"""

def format_input(st):
    return st['system']

IGNORE_INDEX = -100

DEFAULT_PAD_TOKEN = "<|pad|>"
DEFAULT_EOS_TOKEN = "<|endoftext|>"
DEFAULT_UNK_TOKEN = "<|unk|>"

@dataclass
class ModelArguments:
    model_path: Optional[str] = field(
        default="meta-llama/Llama-2-13b-hf")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={
                           "help": "Path to the training data"})
    num_examples: int = field(default=1, metadata={
        "help": "num of examples"
    })

@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch_fused")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum Sequence length"}
    )

        
def smart_tokenizer_and_embedding_resize(special_tokens_dict: Dict, tokenizer: transformers.PreTrainedTokenizer , model: transformers.PreTrainedModel) -> Dict:
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0 , keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0 , keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def _tokenize_fn(strings: Sequence[str] , tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [tokenizer(text , return_tensors="pt" , padding="longest" , max_length=tokenizer.model_max_length,
                                truncation=True
                                ) for text in strings]
    
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    #Get total input length by not including the padding tokens 
    input_id_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_id_lens=input_id_lens,
        labels_lens=labels_lens
    )
    

def preprocess(sources: Sequence[str] , target: Sequence[str] , tokenizer: transformers.PreTrainedTokenizer , dataset_type: str) -> Dict:
    examples = [s + t for s,t in zip(sources , target)]
    example_tokenized , source_tokenized = [_tokenize_fn(strings , tokenizer) for strings in (examples , sources)]
    input_ids = example_tokenized["input_ids"]
    lis = []
    final_labels = []
    labels = copy.deepcopy(input_ids)
    # for i , label , source_len in zip(input_ids , labels , source_tokenized["input_id_lens"]):
    #     label[:source_len] = IGNORE_INDEX
    #     lis.append(i)
    #     final_labels.append(label)
    c = 0
    for ip , label in zip(input_ids , labels):
        temp_label = label.tolist()
        i = 0
        while i < len(temp_label):
            if temp_label[i] == 4816:
                while i < len(temp_label):
                    if (temp_label[i] == 11123 or temp_label[i] ==16368):
                        break
                    i = i + 1
                if i < len(temp_label):
                    temp_label[i] = -100
                if i >= len(temp_label):
                    break
            else:
                temp_label[i] = -100
            i = i + 1
        final_labels.append(torch.tensor(temp_label))
        lis.append(ip)

    print("original dataset size" , len(input_ids))
    print("final dataset size" , len(lis))
    print("number of dropped records" , c)
    
    return dict(input_ids=lis , labels=final_labels)

class SupervisedDataset(Dataset):
    def __init__(self , tokenizer:transformers.PreTrainedTokenizer , main_dataset , dataset_type : str):
        super(SupervisedDataset , self).__init__()
        self.dataset = main_dataset
        targets = []
        sources = []

        for i in self.dataset:
            instruction = format_input(i)
            output = format_output(i) + tokenizer.eos_token
            sources.append(instruction)
            targets.append(output)

        data_dict = preprocess(sources=sources , target=targets , tokenizer=tokenizer , dataset_type=dataset_type)
        
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

        print(f"length of {dataset_type}" , len(self.input_ids))

        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self , i) -> Dict[str , torch.Tensor]:
        return dict(input_ids=self.input_ids[i] , labels=self.labels[i])

@dataclass
class DataCollatorForSuperVisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self , instances: Sequence[Dict]) -> Dict[str , torch.Tensor]:
        input_ids , labels = tuple([instance[key] for instance in instances] for key in ("input_ids" , "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True , padding_value=self.tokenizer.pad_token_id)

        labels = torch.nn.utils.rnn.pad_sequence(labels , batch_first=True , padding_value=IGNORE_INDEX)
        return dict(
            input_ids = input_ids,
            labels = labels,
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        )
    
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer , data_args) ->Dict:
    print("data path" , data_args.data_path)

    main_dataset = load_dataset(data_args.data_path , split="train" , cache_dir = "./")
    main_dataset = main_dataset.train_test_split(test_size=0.02, shuffle=True, seed=42)

    print("train dataset" , main_dataset)
    
    train_data = main_dataset['train']
    eval_data = main_dataset['test']
    
    train_dataset = SupervisedDataset(tokenizer = tokenizer , main_dataset=train_data , dataset_type="Train")
    eval_dataset = SupervisedDataset(tokenizer = tokenizer , main_dataset=eval_data , dataset_type="Eval")
    data_collator = DataCollatorForSuperVisedDataset(tokenizer=tokenizer)

    return dict(train_dataset = train_dataset , eval_dataset = eval_dataset , data_collator = data_collator)




def create_bnb_config():
    """
    bnb config to load the pre trained LM and
    setting computation. 
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return bnb_config

def create_peft_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for the model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
        r=16,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    return config

def find_all_linear_names(model, bits=4):
    cls = bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)



def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


def train():
    parser = transformers.HfArgumentParser((ModelArguments , DataArguments , TrainingArguments))
    model_args , data_args , training_args = parser.parse_args_into_dataclasses()

    print("Model args" , model_args)
    print("data_args" , data_args)
    print("training_args" , training_args)

    #create bnb config
    bnb_config = create_bnb_config()


    model = AutoModelForCausalLM.from_pretrained(
        training_args.cache_dir,
        quantization_config = bnb_config,
        device_map={"": Accelerator().process_index})

    model.config.pretraining_tp = 1

    print("Model is" , model_args.model_path)
    print("max length is" , training_args.model_max_length)
    
    tokenizer = AutoTokenizer.from_pretrained(training_args.cache_dir , padding_side="right" , trust_remote_code=True , model_max_length = training_args.model_max_length)
 

    model = prepare_model_for_kbit_training(model)

    modules = find_all_linear_names(model , 4)
    peft_config = create_peft_config(modules=modules)

    model.enable_input_require_grads()
    model = get_peft_model(model , peft_config)


    print_trainable_parameters(model)

    special_tokens_dict = dict()

    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    

    print("special tokens dict" , special_tokens_dict)
    
    #Resize Embeddings
    smart_tokenizer_and_embedding_resize(special_tokens_dict=special_tokens_dict,
                                         tokenizer=tokenizer,
                                         model= model
                                        )
    
    #Process data
    data_module = make_supervised_data_module(tokenizer=tokenizer , data_args = data_args)


    #Printig One Example from the training data
    print("train dataset Example" , data_module["train_dataset"][0])


    print(tokenizer.decode(data_module["train_dataset"][0]["input_ids"]))
    

    #Create Trainer
    trainer = Trainer(model = model , tokenizer=tokenizer , args=training_args , **data_module)
    trainer.train()

    #trainer.save_state()
    model.save_pretrained(training_args.output_dir)
    #trainer.save_model(output_dir = training_args.output_dir)


if __name__ == "__main__":
    train()