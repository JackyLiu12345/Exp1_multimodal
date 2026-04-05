# Data Format Documentation

This document describes the expected data format for the multimodal fake news detection pipeline.

## Expected JSON Schema

Each split (train, dev, test) is a JSON file containing an array of objects:

```json
[
  {
    "id": "tweet_123",
    "text": "This is the text content of the news post...",
    "image_path": "data/raw/images/tweet_123.jpg",
    "has_image": true,
    "label": "fake"
  },
  {
    "id": "tweet_456",
    "text": "Another news article text.",
    "image_path": null,
    "has_image": false,
    "label": "real"
  }
]
```

## Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier for the sample |
| `text` | string | Yes | The text content of the news article or social media post |
| `image_path` | string \| null | No | Absolute or relative path to the associated image file |
| `has_image` | boolean | No | Whether this sample has an associated image (default: `false`) |
| `label` | string | Yes | Ground truth label: `"real"` or `"fake"` |

## Label Mapping

The project uses a consistent label mapping across all splits:

```python
{"real": 0, "fake": 1}
```

> **Important**: Do not use other label names (e.g., "true"/"false", 0/1 as integers). If your dataset uses different labels, remap them before creating the JSON files.

## Directory Structure

```
data/
├── raw/
│   ├── twitter15/        # Raw Twitter-15 data
│   ├── fakenewsnet/      # FakeNewsNet CSV data
│   │   ├── train.json
│   │   └── test.json
│   └── images/           # Image files referenced by image_path
├── processed/
│   └── twitter15/        # Preprocessed data
│       ├── train.json
│       ├── dev.json
│       └── test.json
└── downloads/            # Cached dataset archives
```

## Preparing Data

### From FakeNewsNet CSV

```bash
python scripts/process_fakenewsnet_csv.py
```

This will produce `data/raw/fakenewsnet/train.json` and `data/raw/fakenewsnet/test.json`.

### From Twitter-15/16

```bash
python scripts/collect_data.py --dataset twitter15 --output_dir data/raw
python scripts/preprocess.py --dataset twitter15 --output_dir data/processed
```

### Custom Datasets

To use a custom dataset, create JSON files matching the schema above and place them in `data/processed/<dataset_name>/` with files named `train.json`, `dev.json`, and `test.json`.

## Image Requirements

- Format: JPEG, PNG, or other PIL-supported formats
- Will be automatically resized to 224×224 pixels
- RGB format (grayscale will be converted)
- Missing or corrupt images are silently replaced with zero tensors

## Example: Minimal Dataset

```json
[
  {"id": "1", "text": "Breaking: major event confirmed", "label": "real", "has_image": false},
  {"id": "2", "text": "SHOCKING revelation that never happened", "label": "fake", "has_image": false}
]
```
