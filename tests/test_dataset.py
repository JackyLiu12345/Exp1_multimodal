#!/usr/bin/env python3
"""
Unit tests for the dataset loader.
"""

import pytest
import json
import torch
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dataset_loader import MultimodalDataset, collate_fn, DEFAULT_LABEL2ID


@pytest.fixture
def sample_data_file(tmp_path):
    """Create a temporary JSON data file for testing."""
    samples = [
        {"id": "1", "text": "This is real news.", "label": "real", "has_image": False},
        {"id": "2", "text": "This is fake news!", "label": "fake", "has_image": False},
        {"id": "3", "text": "Another real article.", "label": "real", "has_image": False},
        {"id": "4", "text": "", "label": "fake", "has_image": False},  # empty text
    ]
    data_file = tmp_path / "test_data.json"
    data_file.write_text(json.dumps(samples))
    return str(data_file)


@pytest.fixture
def unknown_label_data_file(tmp_path):
    """Data file with labels not in the default mapping."""
    samples = [
        {"id": "1", "text": "Sample 1", "label": "true", "has_image": False},
        {"id": "2", "text": "Sample 2", "label": "false", "has_image": False},
    ]
    data_file = tmp_path / "unknown_labels.json"
    data_file.write_text(json.dumps(samples))
    return str(data_file)


class TestMultimodalDataset:
    """Tests for the MultimodalDataset class."""

    def test_loading(self, sample_data_file):
        dataset = MultimodalDataset(sample_data_file)
        assert len(dataset) == 4

    def test_default_label_mapping(self, sample_data_file):
        dataset = MultimodalDataset(sample_data_file)
        assert dataset.label2id == {"real": 0, "fake": 1}

    def test_custom_label_mapping(self, sample_data_file):
        custom = {"real": 1, "fake": 0}
        dataset = MultimodalDataset(sample_data_file, label2id=custom)
        assert dataset.label2id == custom

    def test_getitem_returns_dict(self, sample_data_file):
        dataset = MultimodalDataset(sample_data_file)
        item = dataset[0]
        assert "text_input_ids" in item
        assert "attention_mask" in item
        assert "image" in item
        assert "label" in item
        assert "has_image" in item

    def test_getitem_label_correct(self, sample_data_file):
        dataset = MultimodalDataset(sample_data_file)
        # First sample is "real" -> 0
        item0 = dataset[0]
        assert item0["label"].item() == 0
        # Second sample is "fake" -> 1
        item1 = dataset[1]
        assert item1["label"].item() == 1

    def test_image_tensor_shape(self, sample_data_file):
        dataset = MultimodalDataset(sample_data_file)
        item = dataset[0]
        assert item["image"].shape == (3, 224, 224)

    def test_no_image_returns_zeros(self, sample_data_file):
        dataset = MultimodalDataset(sample_data_file)
        item = dataset[0]
        assert not item["has_image"]
        assert torch.all(item["image"] == 0)

    def test_class_weights(self, sample_data_file):
        dataset = MultimodalDataset(sample_data_file)
        weights = dataset.get_class_weights()
        assert weights.shape == (2,)
        assert (weights > 0).all()

    def test_label_distribution(self, sample_data_file):
        dataset = MultimodalDataset(sample_data_file)
        dist = dataset.get_label_distribution()
        assert dist["real"] == 2
        assert dist["fake"] == 2

    def test_unknown_labels_default_to_zero(self, unknown_label_data_file):
        """Samples with unknown labels should map to 0 by default."""
        dataset = MultimodalDataset(unknown_label_data_file)
        item = dataset[0]
        # "true" is not in default label2id, so label_id defaults to 0
        assert item["label"].item() == 0


class TestCollateFn:
    """Tests for the batch collation function."""

    def _make_tokenizer(self):
        """Create a simple mock tokenizer for testing."""
        class MockTokenizer:
            def __call__(self, text, truncation=True, max_length=512, padding="max_length", return_tensors="pt"):
                # Return fixed-size tensors
                ids = torch.zeros(1, max_length, dtype=torch.long)
                mask = torch.ones(1, max_length, dtype=torch.long)
                return {"input_ids": ids, "attention_mask": mask}
        return MockTokenizer()

    def test_collate_basic(self, sample_data_file):
        tokenizer = self._make_tokenizer()
        dataset = MultimodalDataset(sample_data_file, text_tokenizer=tokenizer)
        items = [dataset[i] for i in range(2)]
        batch = collate_fn(items)
        assert batch["text_input_ids"].shape[0] == 2
        assert batch["image"].shape[0] == 2
        assert batch["label"].shape[0] == 2
        assert len(batch["ids"]) == 2

    def test_collate_preserves_labels(self, sample_data_file):
        tokenizer = self._make_tokenizer()
        dataset = MultimodalDataset(sample_data_file, text_tokenizer=tokenizer)
        items = [dataset[0], dataset[1]]
        batch = collate_fn(items)
        assert batch["label"][0].item() == 0  # real
        assert batch["label"][1].item() == 1  # fake


class TestDefaultLabelMapping:
    """Tests for consistent label mapping."""

    def test_default_mapping(self):
        assert DEFAULT_LABEL2ID == {"real": 0, "fake": 1}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
