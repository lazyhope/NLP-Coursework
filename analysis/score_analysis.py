import pandas as pd
import torch
from sklearn.metrics import classification_report
from transformers import AutoTokenizer

from train.train_freezed import PCLClassifier, predict_text

org_df = pd.read_csv(
    "data/dontpatronizeme_pcl.tsv", sep="\t", skiprows=range(3), header=None
)
id2score = {}
for _, row in org_df.iterrows():
    id2score[row[0]] = row[5]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("roberta-large")

model = PCLClassifier(model_name="roberta-large")
model.load_state_dict(torch.load("checkpoints/best_freezed.pt"))
model = model.to(device)
model.eval()

test_df = pd.read_csv("data/test_data.csv")

test_df["pred"] = test_df["text"].apply(
    lambda x: predict_text(model, x, tokenizer, device)
)
test_df["score"] = test_df["par_id"].map(id2score)

for score in sorted(test_df["score"].unique()):
    score_df = test_df[test_df["score"] == score]
    print("-" * 60)
    title = f"Score: {score}"
    report = classification_report(score_df["label"], score_df["pred"], zero_division=0)
    report = title + report[len(title) :]
    print("\n".join(report.splitlines()[:-4]))
