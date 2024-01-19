model_path=
tp_size=1
dataset=Nexusflow/OTXAPIBenchmark
dataset_split=train
output_file=
dataset_type=otx
nohup python inference.py \
    --model_name $model_path\
    --tp_size $tp_size \
    --dataset $dataset\
    --dataset_split $dataset_split \
    --output_file $output_file \
    --dataset_type $dataset_type > inference_logs.txt
