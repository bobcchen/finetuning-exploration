From https://github.com/facebookresearch/llama-recipes/

specifically from commit 43771602c9d7808c888eb5995ccce4bc8beafb1f as of 1/12/2023

# Model

Download official Llama2 from Meta, specifically the 7b and 13b variants.

Use https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py to convert to hf format

Output folder at and `model-7b` and `model-13b`

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
    --split_size 1% \
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
    --split_size 1% \
    --enable_fsdp \
    --use_peft \
    --quantization False \
    --use_fp16 True \
```


