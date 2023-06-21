from transformers import AutoModelForSeq2SeqLM, AutoTokenizer



print("loading the models")

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")



print("loading the tok")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

inputs = tokenizer("A step by step recipe to make bolognese pasta:", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))