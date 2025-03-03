The given commands should be run under the repository root.

## Training

To train the model, run:
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
