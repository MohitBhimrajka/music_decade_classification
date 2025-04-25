# src/utils.py

import torch
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
PROCESSED_DATA_DIR = "../data/processed/"
TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train.pt")
VAL_PATH = os.path.join(PROCESSED_DATA_DIR, "val.pt")
TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "test.pt")

def load_processed_data():
    """Loads the processed train, validation, and test tensors."""
    try:
        logging.info(f"Loading data from {PROCESSED_DATA_DIR}...")
        X_train, y_train = torch.load(TRAIN_PATH)
        X_val, y_val = torch.load(VAL_PATH)
        X_test, y_test = torch.load(TEST_PATH)
        logging.info("Processed data loaded successfully.")
        logging.info(f"  Train shapes: X={X_train.shape}, y={y_train.shape}")
        logging.info(f"  Val shapes: X={X_val.shape}, y={y_val.shape}")
        logging.info(f"  Test shapes: X={X_test.shape}, y={y_test.shape}")
        return X_train, y_train, X_val, y_val, X_test, y_test
    except FileNotFoundError:
        logging.error(f"Error: Processed data files not found in {PROCESSED_DATA_DIR}. Run data_processing.py first.")
        raise
    except Exception as e:
        logging.error(f"Error loading processed data: {e}")
        raise

if __name__ == '__main__':
    # Example usage: Load data when script is run directly
    load_processed_data()