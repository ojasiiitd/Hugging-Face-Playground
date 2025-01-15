# pipeline tasks - we can also use text-generation etc 
# https://huggingface.co/docs/transformers/quicktour
from transformers import pipeline

# classifier = pipeline("sentiment-analysis")
# res = classifier("yesterday I had meat with my friends")
# print(res)


# under the hood for the pipeline:
from transformers import AutoTokenizer , AutoModelForSequenceClassification , BertTokenizer , BertModel

model_name = 'bert-base-multilingual-cased'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# classifier = pipeline("sentiment-analysis" , model = model , tokenizer = tokenizer)
# res = classifier("yesterday I had meat with my friends")
# print(res)

sequence = "இந்த எடுத்துக்காட்டுகளில், 'அபிமானம்' மற்றும் 'பற்று' ஆகிய இரண்டும் affective தொடர்புகளைக் குறிப்பது."
res = tokenizer(sequence)
print(res)

tokens = tokenizer.tokenize(sequence)
print(tokens)

# finetuning
# https://huggingface.co/docs/transformers/training