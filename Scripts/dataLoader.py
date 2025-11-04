import os
import pandas as pd

def load_raw_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_path, "../data/drugsComTrain_raw.csv")
    test_path = os.path.join(base_path, "../data/drugsComTest_raw.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print("✅ Data loaded successfully!")
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    return train_df, test_df
