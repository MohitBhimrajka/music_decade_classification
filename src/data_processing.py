# src/data_processing.py

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
RAW_DATA_PATH = "../data/raw/YearPredictionMSD.txt"
PROCESSED_DATA_DIR = "../data/processed/"
SCALER_PATH = os.path.join(PROCESSED_DATA_DIR, "scaler.pkl")
TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train.pt")
VAL_PATH = os.path.join(PROCESSED_DATA_DIR, "val.pt")
TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "test.pt")

# Splitting proportions
TEST_SIZE = 0.15  # 15% for testing
VAL_SIZE_RATIO = 0.15 / (1 - TEST_SIZE) # Validation size as a fraction of the non-test set (15% / 85% = ~0.1765)

# Dataset specific info
N_FEATURES = 90

# --- Helper Functions ---

def load_data(filepath):
    """Loads the dataset from the specified txt file."""
    logging.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, header=None)
        # Add column names
        colnames = ['Year'] + [f'Feature_{i+1}' for i in range(N_FEATURES)]
        df.columns = colnames
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        # Verify year range
        min_year, max_year = df['Year'].min(), df['Year'].max()
        logging.info(f"Year range in dataset: {min_year} - {max_year}")
        return df
    except FileNotFoundError:
        logging.error(f"Error: Raw data file not found at {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def create_decade_bins(df):
    """Converts the 'Year' column into decade labels (0-9)."""
    logging.info("Creating decade bins...")
    # Define decade boundaries and labels
    # 1920s (1920-1929) -> 0, ..., 2010s (2010-2019) -> 9
    min_year = 1920 # Start decade reference
    df['Decade_Start'] = (df['Year'] // 10) * 10
    df['Decade_Label'] = ((df['Decade_Start'] - min_year) // 10).astype(int)

    # Ensure labels are within expected range (e.g., handle potential edge cases if needed)
    df['Decade_Label'] = df['Decade_Label'].clip(lower=0) # Clip any year before 1920 to label 0

    # Create string representation for verification/EDA
    decade_map = {i: f"{min_year + i*10}s" for i in range(10)} # Creates {0: '1920s', 1: '1930s', ..., 9: '2010s'}
    df['Decade_Name'] = df['Decade_Label'].map(decade_map)

    logging.info("Decade bins created:")
    logging.info(f"\n{df[['Year', 'Decade_Start', 'Decade_Label', 'Decade_Name']].head()}")
    logging.info(f"\nValue counts for Decade Labels:\n{df['Decade_Label'].value_counts().sort_index()}")
    logging.info(f"\nValue counts for Decade Names:\n{df['Decade_Name'].value_counts().sort_index()}")

    # Check if any song maps to unexpected labels (optional sanity check)
    if df['Decade_Label'].max() > 9:
        logging.warning(f"Warning: Maximum decade label generated is {df['Decade_Label'].max()}, expected 9.")
    if df['Decade_Label'].min() < 0:
         logging.warning(f"Warning: Minimum decade label generated is {df['Decade_Label'].min()}, expected 0.")

    return df

# --- Main Processing Function ---

def process_and_save_data():
    """Loads, processes, splits, scales, and saves the data."""

    df = load_data(RAW_DATA_PATH)
    df = create_decade_bins(df)

    # Separate features (X) and target labels (y)
    X = df.iloc[:, 1:N_FEATURES+1].values # Features start from the second column
    y = df['Decade_Label'].values         # Use the numeric decade label

    # --- Data Splitting (70% train, 15% validation, 15% test) ---
    logging.info("Splitting data into train, validation, and test sets (70/15/15)...")

    # Step 1: Split into Train (70%) and Temp (30% - for Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(VAL_SIZE_RATIO + TEST_SIZE)/(1-TEST_SIZE+VAL_SIZE_RATIO+TEST_SIZE), # Should be 0.30
        random_state=42,
        stratify=y # Stratify based on decade labels
    )

    # Step 2: Split Temp (30%) into Validation (15%) and Test (15%)
    # The split ratio here should be val_size / (val_size + test_size) = 0.15 / (0.15 + 0.15) = 0.5
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5, # 50% of the temp set -> 15% of original total
        random_state=42,
        stratify=y_temp # Stratify based on decade labels
    )

    logging.info(f"Data split complete:")
    logging.info(f"  Train shape: X={X_train.shape}, y={y_train.shape}")
    logging.info(f"  Validation shape: X={X_val.shape}, y={y_val.shape}")
    logging.info(f"  Test shape: X={X_test.shape}, y={y_test.shape}")

    # Verify distributions (optional check)
    train_dist = np.bincount(y_train) / len(y_train)
    val_dist = np.bincount(y_val) / len(y_val)
    test_dist = np.bincount(y_test) / len(y_test)
    logging.info(f"  Train label distribution (normalized): {np.round(train_dist, 3)}")
    logging.info(f"  Validation label distribution (normalized): {np.round(val_dist, 3)}")
    logging.info(f"  Test label distribution (normalized): {np.round(test_dist, 3)}")


    # --- Feature Scaling ---
    logging.info("Scaling features using StandardScaler (fitted on training data only)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --- Convert to PyTorch Tensors ---
    logging.info("Converting data to PyTorch tensors...")
    # Ensure features are float32 and labels are long integers
    train_X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    train_y_tensor = torch.tensor(y_train, dtype=torch.long)
    val_X_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    val_y_tensor = torch.tensor(y_val, dtype=torch.long)
    test_X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    test_y_tensor = torch.tensor(y_test, dtype=torch.long)

    logging.info("Tensor conversion complete.")
    logging.info(f"  Train tensor shapes: X={train_X_tensor.shape}, y={train_y_tensor.shape}")
    logging.info(f"  Val tensor shapes: X={val_X_tensor.shape}, y={val_y_tensor.shape}")
    logging.info(f"  Test tensor shapes: X={test_X_tensor.shape}, y={test_y_tensor.shape}")


    # --- Save Processed Data and Scaler ---
    logging.info(f"Saving processed data to {PROCESSED_DATA_DIR}...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Save tensors
    torch.save((train_X_tensor, train_y_tensor), TRAIN_PATH)
    torch.save((val_X_tensor, val_y_tensor), VAL_PATH)
    torch.save((test_X_tensor, test_y_tensor), TEST_PATH)
    logging.info("Train, validation, and test tensors saved.")

    # Save the scaler
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    logging.info(f"Scaler object saved to {SCALER_PATH}")

    logging.info("Data processing finished successfully.")


# --- Main Execution ---
if __name__ == "__main__":
    process_and_save_data()