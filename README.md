# Music Decade Classification using Deep Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project details the process of building and systematically optimizing a Deep Neural Network (DNN) to classify songs into their release decades (1920s-2010s) based on audio features from the Million Song Dataset subset.

**Read the full journey and detailed findings in our blog post:**
➡️ **[Building an Optimized Deep Neural Network: Lessons from Music Classification](https://mohitbhimrajka.com/blog/music_decade_classification)** ⬅️

## Project Overview

The goal was to take the YearPredictionMSD dataset, originally a regression task for predicting the exact release year, transform it into a decade classification problem, and then explore various DNN optimization techniques. We documented each step, from data preprocessing and exploratory analysis to finding optimal hyperparameters and network components.

**Key Steps & Techniques Explored:**

*   **Data Preparation:** Decade binning, stratified train/validation/test splitting (70/15/15), feature scaling (StandardScaler).
*   **Exploratory Data Analysis (EDA):** Visualizing feature distributions, correlations, class imbalance, and outliers.
*   **Baseline Modeling:** Comparing initial performance of different DNN architectures (Moderate, Wide, Deeper).
*   **Learning Rate Optimization:** Using `torch-lr-finder` to identify an optimal learning rate.
*   **Weight Decay (L2 Regularization):** Employing K-Fold Cross-Validation (k=5) to find the best weight decay value.
*   **Component Optimization:** Systematically testing and comparing:
    *   Weight Initialization (Default, Xavier, Kaiming)
    *   Activation Functions (ReLU, LeakyReLU, GELU)
    *   Normalization Layers (None, BatchNorm, LayerNorm)
    *   Optimizers (Adam, SGD with Momentum, RMSprop)
*   **Final Evaluation:** Training the fully optimized model and evaluating performance on the held-out test set.

## Final Result

After the systematic optimization process, the final model achieved:

*   **Test Accuracy:** **66.05%**
*   **Test Loss:** 0.8993

*(See the [blog post](https://mohitbhimrajka.github.io/music_decade_classification/) for the detailed breakdown of how each optimization step contributed.)*

## Repository Structure

```
├── data/               # Data directory (requires manual download)
│   ├── raw/           # Raw YearPredictionMSD.txt file
│   └── processed/     # Processed .pt tensor files and scaler.pkl
├── docs/               # Contains files for the GitHub Pages blog post
│   ├── assets/        # CSS and Images for the blog
│   ├── _config.yml    # Jekyll configuration
│   └── index.html     # Main blog post content
├── notebooks/         # Jupyter notebooks detailing the workflow
│   ├── 1_Data_Exploration.ipynb
│   ├── 2_Initial_Model_Runs.ipynb
│   └── 3_Optimization_Experiments.ipynb # Main optimization workflow
├── reports/           # Internal report document (Part 1 & 2)
├── results/           # Saved outputs from experiments
│   ├── models/        # Saved final model state (.pth)
│   └── plots/         # Generated plots (.png) used in notebooks/blog
├── src/               # Source code modules
│   ├── data_processing.py # Data loading, cleaning, transformation, splitting
│   ├── models.py        # DNN model definitions (revised for flexibility)
│   └── utils.py         # Utility functions (e.g., loading processed data)
├── .gitignore
├── LICENSE
├── README.md          # This file
└── requirements.txt   # Python dependencies
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MohitBhimrajka/music_decade_classification.git
    cd music_decade_classification
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install torch-lr-finder # If not included in requirements.txt
    ```
4.  **Download the dataset:**
    *   Download `YearPredictionMSD.txt` from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd) or [Kaggle](https://www.kaggle.com/datasets/uciml/millionsongdataset).
    *   Place the downloaded `YearPredictionMSD.txt` file inside the `data/raw/` directory.

## Usage

1.  **Process the Data:** Run the data processing script first. Navigate to the `src` directory and run:
    ```bash
    cd src
    python data_processing.py
    cd ..
    ```
    This will create the processed data files in `data/processed/`.

2.  **Explore the Notebooks:** The primary workflow and experiments are detailed in the Jupyter notebooks located in the `notebooks/` directory. It's recommended to run them in order:
    *   `1_Data_Exploration.ipynb`: Performs EDA on the raw data.
    *   `2_Initial_Model_Runs.ipynb`: Compares baseline model architectures.
    *   `3_Optimization_Experiments.ipynb`: Contains the full optimization pipeline (LR finder, WD K-Fold, component tests, final training, and evaluation).

    You can start Jupyter Lab/Notebook from the project's root directory:
    ```bash
    jupyter lab
    ```
    Then navigate to the `notebooks` folder and run the cells.

## Technologies Used

*   Python 3.x
*   PyTorch
*   Pandas
*   NumPy
*   Scikit-learn
*   Matplotlib & Seaborn
*   Jupyter Notebook / Lab
*   torch-lr-finder
*   Jekyll & GitHub Pages (for the blog post)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
