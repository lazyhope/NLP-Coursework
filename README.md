The given commands should be run under the repository root.

## Download Our best model

To download our best model, run:
```bash
curl -L https://huggingface.co/Lazyhope/roberta-pcl/resolve/main/best_freezed.pt?download=true -o checkpoints/best_freezed.pt
```

## Training

To train the model from scratch, run:
```bash
python -m train.train_freezed
```

To reuse the pre-trained model checkpoint, run:
```bash
python -m train.train_freezed --model checkpoints/best_freezed.pt
```

## Score Analysis

To analyze how the model performs on different scores, run:
```bash
python -m analysis.score_analysis
```

## Predict Dev and Test

To predict the dev and test set, run:
```bash
python predict_dev_test.py
```
And view the results in `dev.txt` and `test.txt`.
