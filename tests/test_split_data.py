import pytest
from swin_transformer.split_data import split_dataset

def test_split_proportions():
    ids = [f"img_{i}" for i in range(100)]
    train, val, test = split_dataset(ids.copy(), train_frac=0.7, val_frac=0.2, test_frac=0.1)

    assert len(train) == 70, "❌ Train size incorrect"
    assert len(val) == 20, "❌ Validation size incorrect"
    assert len(test) == 10, "❌ Test size incorrect"

def test_all_ids_preserved():
    ids = [f"img_{i}" for i in range(50)]
    train, val, test = split_dataset(ids.copy())

    combined = train + val + test
    assert sorted(combined) == sorted(ids), "❌ IDs lost or duplicated"

def test_small_dataset():
    ids = [f"img_{i}" for i in range(3)]
    train, val, test = split_dataset(ids.copy(), train_frac=0.6, val_frac=0.2, test_frac=0.2)

    assert len(train) + len(val) + len(test) == 3, "❌ Total samples don't match input size"

def test_100_percent_train():
    ids = [f"img_{i}" for i in range(10)]
    train, val, test = split_dataset(ids.copy(), train_frac=1.0, val_frac=0.0, test_frac=0.0)

    assert len(train) == 10
    assert len(val) == 0
    assert len(test) == 0

def test_split_sum_equals_original():
    ids = [f"img_{i}" for i in range(25)]
    train, val, test = split_dataset(ids.copy())
    total = len(train) + len(val) + len(test)

    assert total == len(ids), "❌ Total split size != original size"
