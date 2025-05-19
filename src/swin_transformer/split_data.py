import numpy as np

def split_dataset(ids, train_frac=0.7, val_frac=0.1, test_frac=0.2):
    # Sort ids to ensure consistent ordering across runs
    ids = sorted(ids)
    
    total_ids = len(ids)
    train_end = int(total_ids * train_frac)
    val_end = train_end + int(total_ids * val_frac)
    
    train_ids = ids[:train_end]
    val_ids = ids[train_end:val_end]
    test_ids = ids[val_end:]
    
    return train_ids, val_ids, test_ids