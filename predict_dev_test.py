import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import classification_report

from train.train_freezed import PCLClassifier, predict_text

dev_df = pd.read_csv("data/test_data.csv")
test_df = pd.read_csv("data/task4_test.tsv", sep="\t", header=None)

dev_texts = dev_df["text"]
test_texts = test_df[4]

tokenizer = AutoTokenizer.from_pretrained("roberta-large")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PCLClassifier(model_name="roberta-large")
model.load_state_dict(torch.load("checkpoints/best_freezed.pt"))
model = model.to(device)

dev_preds = []
with open("dev.txt", "w") as f:
    for text in tqdm(dev_texts):
        pred = predict_text(model, text, tokenizer, device)
        dev_preds.append(pred)
        f.write(f"{pred}\n")
print(classification_report(dev_df["label"], dev_preds))

with open("test.txt", "w") as f:
    for text in tqdm(test_texts):
        pred = predict_text(model, text, tokenizer, device)
        f.write(f"{pred}\n")
