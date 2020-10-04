import transformers


tokenizer = transformers.AutoTokenizer.from_pretrained("../berty-stuff/")
model = transformers.AutoModel.from_pretrained("../berty-stuff/")

