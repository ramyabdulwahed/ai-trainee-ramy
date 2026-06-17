import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel #pEFT is the library that handles LoRA adapters. PeftModel lets us do: base model + adapter

base = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16, #FP16 means 16-bit floating point. Why FP16? Because when merging, we want a normal full-precision-ish model, not the 4-bit training version. During QLoRA training, the base model was loaded in 4-bit to save GPU memory. But for merging, we usually reload the base model in FP16 and then apply the LoRA adapter into it.
    device_map="cpu"
)

#take the base model and load my trained LoRA adapter from OUT. then bake the adapter weights into the base model weights, then remove the separate adapter structure.
merged = PeftModel.from_pretrained(base, OUT).merge_and_unload()

#save merged model
merged.save_pretrained("models/day09_merged_fp16")

#saves the tokenizer into the same folder, because a model folder is not complete without the tokenizer.
AutoTokenizer.from_pretrained(MODEL).save_pretrained("models/day09_merged_fp16")
