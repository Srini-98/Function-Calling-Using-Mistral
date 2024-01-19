# FSDP Training

The Mistral 7B is trained using FSDP with 4 A100 GPU's. The batch size is set to 4 as a higher batch size led to OOM's. Due to time and compute constraints the model was trained for 1 epoch. ( I expect the model to do even better with another epoch of training).

Before starting the training job set your model path , output directory and other parameters in [train.py](https://github.com/Srini-98/Function-Calling-Using-Mistral/blob/master/FSDP/train.py).  The command to start the training job is:
```
torchrun --nnodes=1 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 train.py
```
## Dataset
The same dataset that was used for q lora training was used for full finetuning. ([Link here](glaiveai/glaive-function-calling-v2))

## Inference and evaluation
The vllm library is used for inferencing on the evaluation dataset. Refer to this [Eval section](https://github.com/Srini-98/Function-Calling-Using-Mistral?tab=readme-ov-file#evaluation) for how the model outputs are generated. 

The inference can be done using the following code: (Note you have to set the values in the bash script)

```
bash run_inference.sh
```

Few custom self defined functions for other project needs were also defined and a 'vibe check' of the model was done where it performed well.

### Results
The model comes out with an accuracy of 78%. The outputs of the model can be found here and the script for evaluation is given here [Link here](https://github.com/Srini-98/Function-Calling-Using-Mistral/blob/master/FSDP/eval.py)

### Model Weights

The fully finetuned model weights are available on huggingface. [Model Weights](https://huggingface.co/srini98/mistral-function-calling/tree/main)
