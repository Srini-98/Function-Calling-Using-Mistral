import random 

from core.supervised_dataset_function import (
    DEFAULT_EOS_TOKEN,
    DEFAULT_UNK_TOKEN,
    DEFAULT_PAD_TOKEN,
    SuperVisedDataset,
    DataCollatorForSuperVisedDataset
)

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP , 
    MixedPrecision , 
    FullStateDictConfig,
    StateDictType
)
import datasets
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
import functools
import torch.distributed as dist
import wandb
import uuid
import torch
import transformers
import os 
import math 
import numpy as np 
from datetime import datetime
from transformers import AutoModelForCausalLM , AutoTokenizer

def setup_model(model_name , max_length):
    #config = transformers.AutoConfig.from_pretrained(model_name)
    #config.use_cache = False

    print("model name" , model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name ,
                                                torch_dtype=torch.bfloat16
                                                )
    tokenizer = AutoTokenizer.from_pretrained(model_name , model_max_length=max_length,
                                                           padding_side='right',
                                                           pad_token = DEFAULT_PAD_TOKEN
                                                           )
    
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict['eos_token'] = DEFAULT_EOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict['unk_token'] = DEFAULT_UNK_TOKEN
    
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    return model , tokenizer

def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor , op=torch.distributed.ReduceOp.SUM)
    tensor /= torch.distributed.get_world_size()
    return tensor

def evaluation(model , eval_dataloader , wandb , local_rank):
    if local_rank == 0:
        print("Starting evaluation")
    
    model.eval()
    losses = 0
    for step , batch in enumerate(eval_dataloader):
        inputs = {
            "input_ids": batch['input_ids'].to(model.device),
            "attention_mask": batch['attention_mask'].to(model.device),
            "labels": batch['labels'].to(model.device)
        }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs.loss
        losses += loss.float()

    losses = losses /(step + 1)
    val_loss = get_all_reduce_mean(losses.clone()).item()

    if local_rank == 0:
        wandb.log(
            {
                "eval_loss": val_loss
            }
        )
    return val_loss


def get_dataloader(max_length , world_size , dataset , local_rank , shuffle , seed , collator , batch_size):

    sampler = DistributedSampler(
        dataset , num_replicas=world_size , rank=local_rank  , seed=seed
    )

    loader = DataLoader(
        dataset, pin_memory=True , sampler=sampler , collate_fn=collator , batch_size=batch_size
    )

    return sampler , loader

def get_parameter_names(model , forbidden_layer_types):
    result = []
    for name , child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child , forbidden_layer_types)
            if not isinstance(child , tuple(forbidden_layer_types))
        ]
    result += list(model._parameters.keys())
    return result

def get_optimizer(model , lr , weight_decay):
    decay_parameters = get_parameter_names(model , [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    optimizer_grouped_parameters = [
        {   "params": [
                p for n,p in model.named_parameters()
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n,p in model.named_parameters()
                       if (n not in decay_parameters and p.requires_grad)
                       ],
            "weight_decay": 0.0,
        }
    ]

    return torch.optim.AdamW(
        params=optimizer_grouped_parameters , lr=lr , betas=(0.9 , 0.95) , eps=1e-8,
        weight_decay=weight_decay
    )

def should_run_eval(total_steps , times_to_run , current_step):
    return current_step % (total_steps // times_to_run) == 0

def log_stats(pbar , wandb , epoch , loss_tensor , grad_norm , scheduler):
    last_lr = scheduler.get_last_lr()[0]

    wandb.log(
        {
            "current_loss": loss_tensor,
            "current_epoch": epoch,
            "learning_rate": last_lr,
            "grad_norm": grad_norm
        }
    )

    current_loss = f"{loss_tensor:.4f}"
    current_lr = f"{last_lr:.10f}"

    pbar.set_description(f"Epoch {epoch:.2f}, Loss: {current_loss}, LR: {current_lr}")

def get_warmup_steps(num_training_steps , warmup_ratio=0.05):
    return math.ceil(num_training_steps * warmup_ratio)

def clip_model_gradients(model , max_grad_norm):
    return model.clip_grad_norm_(max_grad_norm).item()

def get_scheduler(local_rank , scheduler_type , optimizer , max_steps):
    warmup_steps = get_warmup_steps(max_steps)

    if local_rank == 0:
        print(f"WARMUP STEPS: {warmup_steps}")
        print(f"MAX STEPS: {max_steps}")
        print(f"SCHEDULER TYPE: {scheduler_type}")
    
    return transformers.get_scheduler(
        name=scheduler_type , 
        optimizer=optimizer ,
        num_warmup_steps=warmup_steps ,
        num_training_steps=max_steps
    )

def save_model(local_rank , model , tokenizer , outpath , current_epoch , current_step):
    save_policy = FullStateDictConfig(offload_to_cpu=True , rank0_only=True)
    with FSDP.state_dict_type(model , StateDictType.FULL_STATE_DICT , save_policy):
        cpu_state = model.state_dict()

    if local_rank == 0:
        print(f"SAVING MODEL")
        outpath += f"/epoch_{current_epoch}_step_{current_step}"
        model.save_pretrained(outpath , state_dict=cpu_state)
        tokenizer.save_pretrained(outpath)

def disable_model_dropout(model):
    for module in model.modules():
        if isinstance(module , torch.nn.Dropout):
            module.p = 0.0

if __name__ == "__main__":
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)

    model_name = "/data/projects/11003644/srini/models/Mistral-7B-v0.1"
    scheduler_type = "cosine"
    seed = 42
    transformers.set_seed(42)
    run_id = str(uuid.uuid4())
    output_dir = f"./function_calling_weights/"
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I_%M_%S_%p")
    max_length = 4096
    disable_dropout = False 
    gradient_checkpointing = True
    clip_gradients = True
    shuffle = True
    train_batch_size = 1
    eval_batch_size = 1
    epochs=1
    acc_steps = 2
    lr=1e-5
    weight_decay=0.01
    gradient_clipping=1.0
    train_on_inputs = False


    model , tokenizer = setup_model(model_name , max_length)
    num_params = sum([p.numel() for p in model.parameters()])
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            MistralDecoderLayer
        },
    )

    fsdp_config = dict(
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy = ShardingStrategy.FULL_SHARD,
        device_id = torch.cuda.current_device(),
        mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
        ),
        backward_prefetch=None,
        param_init_fn = None , 
        cpu_offload=None
    )

    model = FSDP(model , **fsdp_config)
    optimizer = get_optimizer(model , lr , weight_decay)

    train_ds = "glaiveai/glaive-function-calling-v2"
    dataset = datasets.load_dataset(train_ds, split="train")
    main_dataset = dataset.train_test_split(test_size=0.05 , seed=seed)

    if local_rank == 0:
        print("dataset split" , main_dataset)

    train_dataset = SuperVisedDataset(train_on_inputs , tokenizer , main_dataset['train'])
    eval_dataset = SuperVisedDataset(train_on_inputs , tokenizer , main_dataset['test'])
    
    
    collator = DataCollatorForSuperVisedDataset(tokenizer)

    train_sampler , train_dataloader = get_dataloader(
        max_length=max_length , 
        world_size=world_size ,
        dataset=train_dataset ,
        local_rank=local_rank,
        shuffle=shuffle,
        seed=seed,
        collator=collator,
        batch_size=train_batch_size
        )
    
    #WRITE VAL HERE. 
    eval_sampler , val_loader = get_dataloader(
        max_length=max_length,
        world_size=world_size,
        dataset=eval_dataset,
        local_rank=local_rank,
        shuffle=False,
        seed=seed,
        collator=collator,
        batch_size=eval_batch_size
    )
    total_steps_per_epoch = len(train_dataloader)

    max_steps = total_steps_per_epoch * epochs
    scheduler = get_scheduler(local_rank=local_rank , scheduler_type=scheduler_type , optimizer=optimizer , max_steps=max_steps)

    if local_rank == 0:
        run = wandb.init(
            project="mistral-full-finetuning",
            name='fsdp_function_call',
            config={
                "model_name": model_name,
                "dataset_size": len(train_dataset),
                "weight_decay": weight_decay,
                "learning_rate": lr,
                "clip_gradients": clip_gradients,
                "learning_rate": lr,
                "epochs": epochs,
                "batch_size": train_batch_size,
                "total_batch_size": train_batch_size * world_size,
                "scheduler_type": scheduler_type,
                "train_on_inputs": train_on_inputs,
            }
        )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if disable_dropout:
        disable_model_dropout(model)

    model.train()
    dist.barrier()

    for epoch in range(0 , epochs):
        train_sampler.set_epoch(epoch)
        current_epoch = epoch + 1

        pbar = tqdm(
            enumerate(train_dataloader),
            total = total_steps_per_epoch,
            colour="blue",
            desc=f"Epoch {current_epoch:.2f}",
            disable=(local_rank!=0)
        )

        flag = 0
        for step , batch in pbar:
            current_step = step + 1
            inputs  = {
                "input_ids": batch['input_ids'].to(model.device),
                "attention_mask": batch['attention_mask'].to(model.device),
                "labels": batch['labels'].to(model.device)
            }
            if flag == 0:
                if local_rank == 0:
                    print(tokenizer.decode(batch['input_ids'][0]))
                    flag = 1
            
            outputs = model(**inputs)
            loss = outputs.loss

            loss.backward()

            if clip_gradients:
                grad_norm = clip_model_gradients(model , gradient_clipping)
            
            optimizer.step()
            scheduler.step()

            optimizer.zero_grad(set_to_none=True)

            loss = get_all_reduce_mean(loss).item()

            if local_rank == 0:
                log_stats(pbar , wandb , round((current_step / total_steps_per_epoch), 2) + epoch , loss , grad_norm , scheduler)


            if should_run_eval(total_steps_per_epoch , 3 , current_step):
                validation_loss = evaluation(model , val_loader , wandb , local_rank)
            
                save_model(
                    local_rank,
                    model ,
                    tokenizer , 
                    output_dir,
                    current_epoch,
                    current_step
                )

                model.train()

    #save final model
    save_model(local_rank, model, tokenizer, output_dir, epochs, "final")      
