# VORQ Colab Training

This folder contains Google Colab notebooks for GPU-intensive training tasks.

> ⚠️ **Do NOT run these locally.** Open them in [Google Colab](https://colab.research.google.com/) where a free T4/A100 GPU is available.

## Notebooks

| Notebook | Description | GPU Needed |
|----------|-------------|-----------|
| `01_train_event_extractor.ipynb` | Fine-tunes DistilBERT on GDELT event classification | T4 recommended |

## How to Use

1. Upload the notebook to Google Colab (or open via GitHub)
2. Set runtime to **GPU** (Runtime → Change runtime type → GPU)
3. Run all cells
4. Download trained model weights to `vorq/models/event_extractor/`

## After Training

Copy trained model weights back to your local machine and place in:
```
VORQ/
└── vorq/
    └── models/
        └── event_extractor/    ← put model here
            ├── config.json
            ├── pytorch_model.bin
            └── tokenizer_config.json
```

Then the `EventExtractor` in `notebooks/02_engine_core.ipynb` will automatically use the fine-tuned model for higher accuracy.
