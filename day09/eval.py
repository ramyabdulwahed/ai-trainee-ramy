from transformers import AutoTokenizer, AutoModelForCausalLM

from peft import PeftModel #to load the LoRA adapter
import torch, evaluate

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
OUT = "models/day09_llama32_1b_oa_lora"

tok = AutoTokenizer.from_pretrained (MODEL) #We need it for prompts. It turns them into token IDs the model can read

base = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)

#This loads another copy of the base model that gets the adapter attached
tuned = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)

#Take the base model. Load the trained LoRA adapter from OUT. Combine them. So after this line: tuned_model = Llama + trained LoRA adapter
tuned_model = PeftModel.from_pretrained(tuned, OUT)

rouge = evaluate.load("rouge") #loads ROUGE metric used to compare generated text to reference text.
preds,refs =[], []

for ex in eval_ds.select(range(50)): #This takes the first 50 examples from eval_ds ds
    prompt = ex["text"].split("assistant")[0] #take the part before the word "assistant". So prompt becomes the input we give to the model.

    ref = ex["text"].split("assistant", 1)[1] #ref is what we compare the model’s answer against

    #tokenization

    #turns the prompt text into PyTorch tensors, .input_ids gets just the token IDs. Then: .cuda() moves them to the GPU.
    input_ids = tok (prompt, return_tensors="pt").input_ids.cuda()
    out = tuned_model.generate(input_ids, max_new_tokens=200, do_sample=False)
    #out[0] is the full generated token sequence.But we only want the new answer. So: input_ids.shape[-1] gets the length of the prompt. 
    # Then: out[0][input_ids.shape[-1]:] means: Remove the prompt tokens. Keep only the generated tokens. Then: tok.decode(..., skip_special_tokens=True) turns token IDs back into readable text.
    pred = tok.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
    preds.append(pred)
    refs.append(ref)