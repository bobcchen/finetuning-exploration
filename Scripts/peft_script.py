import torch
import transformers
from transformers import BitsAndBytesConfig, LlamaTokenizer, LlamaForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import evaluate
import numpy as np


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def compute_perplexity(model, tokenizer, predictions, batch_size: int = 16, add_start_token: bool = True, device=None,
                       max_length=None):
    # from https://huggingface.co/spaces/evaluate-metric/perplexity/blob/main/perplexity.py

    # if batch_size > 1 (which generally leads to padding being required), and
    # if there is not an already assigned pad_token, assign an existing
    # special token to also be the padding token
    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (
                len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:
        # leave room for <BOS> token to be added:
        assert (
                tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encodings = tokenizer(
        predictions,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    ppls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    for start_index in evaluate.logging.tqdm(range(0, len(encoded_texts), batch_size)):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


def main():
    # create quantization config
    print('Creating quantization config...')
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # use normalized 4bit float for QLoRA
        bnb_4bit_compute_dtype=torch.float16  # can use torch.bfloat16 on newer gpus
    )

    # load model and tokenizer
    print('Loading model and tokenizer...')
    model_path = 'model-7b/'

    model = LlamaForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config, device_map='auto',
    )
    model.config.use_cache = False  # for training
    model.config.pretraining_tp = 1

    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f'initial model device map: {model.hf_device_map}')

    # create lora
    print('Creating LoRA...')
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)

    print(f'model device map after adding lora: {model.hf_device_map}')
    print_trainable_parameters(model)

    # load dataset
    print('Loading dataset...')
    dataset_path = 'dataset/english_quotes.jsonl'

    data = load_dataset("json", data_files=dataset_path, split='train[:20%]')  # use only 20%
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
    data = data.train_test_split(test_size=0.5)  # use 10% for training due to OOM on 16gb gpu; 10% for eval on cpu
    train_data = data['train'].with_format("torch", device='cuda')
    eval_strings = [s for s in data['test']['quote'] if s != '']

    # calculate perplexity before training
    print('Calculating perplexity before training...')
    original_perplexity = compute_perplexity(model, tokenizer, predictions=eval_strings, device='cuda')

    # train model
    print('Training...')
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4, # effective batch size of 1x4 = 4, hence ~60 steps required for 250 training samples
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="./outputs",
            optim="paged_adamw_8bit",
            skip_memory_metrics=False,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    result = trainer.train()

    # calculate perplexity after training
    print('Calculating perplexity after training...')
    perplexity = compute_perplexity(model, tokenizer, predictions=eval_strings, device='cuda')

    # save metrics
    print('Saving metrics...')
    manual_gpu_usage = {
        f'gpu_{device_id}_usage_in_gb': torch.cuda.max_memory_allocated(device=device_id) / 1024 / 1024 / 1024 for
        device_id in range(torch.cuda.device_count())
    }
    result.metrics.update(manual_gpu_usage)

    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)

    eval_metrics = {
        'original_perplexity': original_perplexity['mean_perplexity'],
        'perplexity': perplexity['mean_perplexity'],
    }
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)


if __name__ == "__main__":
    main()
