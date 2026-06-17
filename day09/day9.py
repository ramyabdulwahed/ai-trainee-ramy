from datasets import load_dataset

ds = load_dataset("timdettmers/openassistant-guanaco")

print (ds)

print(ds["train"][0]["text"][:500])