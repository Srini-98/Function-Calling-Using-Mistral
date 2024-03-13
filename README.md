# Function-Calling-Using-Mistral
This repository contains the finetuning and inference code for instruction tuning the Mistral 7B model to do efficient function calling using a chat format.

You can setup the environment using (python>=3.10):
```
pip install -r requirements.txt
```
## QLORA

### Dataset Used:
glaiveai/glaive-function-calling-v2 : https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2

The training split of the dataset was split into training and  development set. The development set is used for evaluation during training and the checkpoint with the lowest evaluation loss is selected as the best checkpoint. 

#### Statistics:

Training Dataset: 110700 examples <br>
Eval Dataset: 2260 examples <br>

#### Dataset Preprocessing:
The loss for the back propagation during finetuning is calculated only on the **Assistant** responses. The labels are accordingly set to -100 (-100 are set for tokens that have to be ignored during loss calculations) based on how the tokenizer of the mistral model tokenizes the sentences. The implementation can be seen below: <br>  https://github.com/Srini-98/Function-Calling-Using-Mistral/blob/59f120cf0a0ef4b21c4893fce93c5b56d3be02e3/train.py#L99

**Note that this could be optimized further but had to move fast!**

## Training Job

The model is fine tuned using the Q lora method where the base model is loaded in 4 bit quantization and adapters are added. The Lora adapaters are added to all the layers except the embeddings. <br>
<br>
A bash script is used to start the training job. The parameters based on the compute cluster has to be set up in the bash script. The job can be started using: 

```
bash run.sh
```

### Model Weights and merging

The lora adapters can be found in the link below:<br>

https://huggingface.co/srini98/Mistral-Function-Calling/tree/main 

Use the command below to merge the weights:

```
python convert_weights.py --model_path {ENTER MODEL PATH} --adapter_path {ENTER ADAPTER PATH} --save_path {ENTER SAVE PATH}
```

## Inference
For Inference the model has to be merged with the lora adapaters that are saved from the finetuning process. 

inference prompt has to be set up in the following format:(replace the function names , description , parameters as per your use case. You can choose to add multiple functions as well. Refer to a sample prompt below or check this eval prompt. ([eval_prompt](custom_function.txt)). You have to parse the output and call the function and append the response into the prompt and let the model generate the final answer. The model's response will always stop with the <|endoftext|> token (use this during inference to stop generation for multiturn inference/function calling)
```
SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -
{
    "name": "calculate_tax",
    "description": "Calculate the tax amount",
    "parameters": {
        "type": "object",
        "properties": {
            "income": {
                "type": "number",
                "description": "The income amount"
            }
        },
        "required": [
            "income"
        ]
    }
}
USER: Hi, I need to calculate my tax for this year. My income is $70,000.


ASSISTANT: Sure, I can help with that. Let me calculate it for you. <|endoftext|>


ASSISTANT: <functioncall> {"name": "calculate_tax", "arguments": '{"income": 70000}'} <|endoftext|>


FUNCTION RESPONSE: {"tax_amount": 17500}


ASSISTANT: Based on your income, your tax for this year is $17,500. <|endoftext|>

```

## Evaluation
For evaluation , a out of distriubtion dataset from nexus flow is used: https://huggingface.co/Nexusflow . The dataset has to be converted into the prompt formart used in training for optimal performance. 

Refer to [eval_prompt](custom_function.txt) for the main prompt with the function definitions. 

## Full Finetuning - Fully Sharded Data Parallel(FSDP)

The model was also fully finetuned using FSDP. The code , inference and evaluation files are present in [FSDP](https://github.com/Srini-98/Function-Calling-Using-Mistral/tree/master/FSDP)
