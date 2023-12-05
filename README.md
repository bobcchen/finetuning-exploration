From https://github.com/facebookresearch/llama-recipes/

specifically from commit 43771602c9d7808c888eb5995ccce4bc8beafb1f as of 1/12/2023

# Model

Download official Llama2 from Meta, specifically the 7b and 13b variants.

Use https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py to convert to hf format

Output folder at `model-7b` and `model-13b`

# Dataset

Download `samsun` dataset to `dataset`

```
from datasets import load_dataset

for split in ['train', 'validation']:
    load_dataset("samsum", split=split).to_json(f'../dataset/samsun_{split}.jsonl')
````
# Runs logging

## Single V100 (32GB) GPU

LoRA PEFT of quantized 7b (`load_in_4bit=True`) in half precision works as per local

```
python3 finetuning.py \
    --split_slice 1% \
    --use_peft \
    --quantization True \
    --use_fp16 True
```

LoRA PEFT of 7b in full/half precision results in CUDA OOM

```
python3 finetuning.py \
    --split_slice 1% \
    --use_peft \
    --quantization False \
    --use_fp16 False
```

```
python3 finetuning.py \
    --split_slice 1% \
    --use_peft \
    --quantization False \
    --use_fp16 True
```

Full finetuning of 7b in full/half precision results in CUDA OOM

```
python3 finetuning.py \
    --split_slice 1% \
    --quantization False \
    --use_fp16 False
```
```
python3 finetuning.py \
    --split_slice 1% \
    --quantization False \
    --use_fp16 True
```

Full finetuning of quantized 7b (`load_in_4bit=True`) in full precision works, but training loss does not decrease

```
python3 finetuning.py \
    --split_slice 1% \
    --quantization True \
    --use_fp16 False 
```

Full finetuning of quantized 7b (`load_in_4bit=True`) in half precision results in

`AssertionError: No inf checks were recorded for this optimizer.`

https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation


```
python3 finetuning.py \
    --split_slice 1% \
    --quantization True \
    --use_fp16 True 
```

## Single node multi V100 (32GB) GPU

FSDP is used to do sharding of model (and data) across gpus.

~~There was a weird issue of the device not setting properly for FSDP, resulting in `Inconsistent compute_device and device_id`, possibly fix it with `export CUDA_VISIBLE_DEVICES=0,1,2,3`.~~ It is due to `quantization` set to `True`, resulting in `device_map` to be `auto`, which maps the model to device 0 and is possibly a bug in the code. Do not use quantization when using FSDP, it is not supported anyway.

Full finetuning of 7b in half precision results in CUDA OOM

```
torchrun \
    --nnodes 1 \
    --nproc_per_node 4 \
    finetuning.py \
    --split_slice 1% \
    --enable_fsdp \
    --quantization False \
    --use_fp16 True \
```

LoRA PEFT of 7b in half precision works! Note that quantization is not supported for FSDP.

eval_ppl and eval_loss decreases similarly to the single gpu case as well.

```
torchrun \
    --nnodes 1 \
    --nproc_per_node 4 \
    finetuning.py \
    --split_slice 1% \
    --enable_fsdp \
    --use_peft \
    --quantization False \
    --use_fp16 True \
```

For FSDP, use 

- `quantization=False` as quantization is not supported for FSDP
~~- `use_fp16=False` in other words full precision to reduce memory usage~~ Proven wrong

However, 

Full finetuning of 7b in full precision results in CUDA OOM, even decreasing samples to 10 `split_slice=10` results in CUDA OOM

Model is successfully loaded across different GPUs (~10GB used per GPU), but when training starts CUDA OOM occurs, possibly due to the optimizer memory usage.

```
torchrun \
    --nnodes 1 \
    --nproc_per_node 4 \
    finetuning.py \
    --split_slice 1% \
    --enable_fsdp \
    --quantization False \
    --use_fp16 False \
```

Looking into FSDP parameters,

- Sharding strategy by default is `FULL_SHARD` which shards all model parameters, optimizer and gradient states, which uses the least GPU memory
- CPU offload (by default false) can be used to offload parameters to CPU when not involved in computation, saving GPU memory at the cost of CPU memory and speed

Full finetuning of 7b in full precision with FSDP CPU_OFFLOAD works but results in OOM (killed by OS)

- CPU Utilization: From logs, CPU total peak memory consumed is 40GB/63GB. From Grafana, ~8 cores used
- GPU Utilization: From logs, GPU total peak memory is 17GB/18GB. From `nvidia-smi`, ~20GB used per GPU

```
torchrun \
    --nnodes 1 \
    --nproc_per_node 4 \
    finetuning.py \
    --split_slice 1% \
    --enable_fsdp \
    --quantization False \
    --use_fp16 False \
    --fsdp_config.fsdp_cpu_offload True \
```

Trains successfully with 10 samples.

```
torchrun \
    --nnodes 1 \
    --nproc_per_node 4 \
    finetuning.py \
    --split_slice 10 \
    --enable_fsdp \
    --quantization False \
    --use_fp16 False \
    --fsdp_config.fsdp_cpu_offload True \
```

After switching to SGD which uses less GPU Memory than AdamW,

Full finetuning of 7b in full precision works, but there are CUDA Malloc retries

- CPU Utilization: From logs, CPU total peak memory consumed is 4GB. From Grafana, ~8 cores used
- GPU Utilization: From logs, GPU total peak memory is 29GB. From `nvidia-smi`, ~30GB used per GPU, which fluctuates between 26-32GB (limit) possibly due to the mallocs
- CUDA Malloc retries ~32 per epoch

```
torchrun \
    --nnodes 1 \
    --nproc_per_node 4 \
    finetuning.py \
    --split_slice 1% \
    --enable_fsdp \
    --quantization False \
    --use_fp16 False \
    --fsdp_config.optimizer SGD \
```

Full finetuning of 7b in half precision works with no issues!

- CPU Utilization: From logs, CPU total peak memory consumed is 4GB. From Grafana, ~8 cores used
- GPU Utilization: From logs, GPU total peak memory is 25GB. From `nvidia-smi`, ~28GB used per GPU

```
torchrun \
    --nnodes 1 \
    --nproc_per_node 4 \
    finetuning.py \
    --split_slice 1% \
    --enable_fsdp \
    --quantization False \
    --use_fp16 True \
    --fsdp_config.optimizer SGD \
```
