# Music Decade Classification using Deep Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains my implementation of a Deep Neural Network (DNN) that classifies songs by their release decade (1920s–2010s), using audio features from the Million Song Dataset subset. I’ve systematically documented the process, from initial data handling to extensive model optimization.

**Read the full journey and detailed findings in my blog post:**  
➡️ **[Building an Optimized Deep Neural Network: Lessons from Music Classification](https://mohitbhimrajka.com/blog/music_decade_classification)** ⬅️

---

## Project Overview

The main goal for this project was to transform the YearPredictionMSD dataset—originally designed for predicting the exact release year—into a decade-based classification task. I explored multiple deep learning techniques, documented every step, and focused on systematic optimization for robust results.

### Key Steps & Techniques

- **Data Preparation:** Decade binning, stratified train/validation/test splitting (70/15/15), and feature scaling (StandardScaler).
- **Exploratory Data Analysis (EDA):** Visualization of feature distributions, correlation analysis, class imbalance inspection, and outlier detection.
- **Baseline Modeling:** Initial performance comparison using different DNN architectures (moderate, wide, deeper).
- **Learning Rate Optimization:** Used `torch-lr-finder` to identify an optimal learning rate for training.
- **Weight Decay (L2 Regularization):** Employed K-Fold Cross-Validation (k=5) to tune the weight decay parameter.
- **Component Optimization:** Systematically tested and compared:
    - Weight Initialization (Default, Xavier, Kaiming)
    - Activation Functions (ReLU, LeakyReLU, GELU)
    - Normalization Layers (None, BatchNorm, LayerNorm)
    - Optimizers (Adam, SGD with Momentum, RMSprop)
- **Final Evaluation:** Trained the fully optimized model and evaluated performance on the held-out test set.

---

## Final Results

After completing the systematic optimization process, my final model achieved:

- **Test Accuracy:** **66.05%**
- **Test Loss:** 0.8993

For a breakdown of how each optimization step contributed to the results, see the [blog post](https://mohitbhimrajka.com/blog/music_decade_classification).

---

## Repository Structure

```
├── data/               # Data directory (requires manual download)
│   ├── raw/           # Raw YearPredictionMSD.txt file
│   └── processed/     # Processed .pt tensor files and scaler.pkl
├── docs/               # Files for the GitHub Pages blog post
│   ├── assets/        # CSS and Images for the blog
│   ├── _config.yml    # Jekyll configuration
│   └── index.html     # Main blog post content
├── notebooks/         # Jupyter notebooks detailing the workflow
│   ├── 1_Data_Exploration.ipynb
│   ├── 2_Initial_Model_Runs.ipynb
│   └── 3_Optimization_Experiments.ipynb
├── reports/           # Internal report documents
├── results/           # Saved outputs from experiments
│   ├── models/        # Saved final model state (.pth)
│   └── plots/         # Generated plots (.png) used in notebooks/blog
├── src/               # Source code modules
│   ├── data_processing.py   # Data loading, cleaning, transformation, splitting
│   ├── models.py            # DNN model definitions (revised for flexibility)
│   └── utils.py             # Utility functions (e.g., loading processed data)
├── .gitignore
├── LICENSE
├── README.md          # This file
└── requirements.txt   # Python dependencies
```

---

## Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/MohitBhimrajka/music_decade_classification.git
    cd music_decade_classification
    ```

2. **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install torch-lr-finder  # If not included in requirements.txt
    ```

4. **Download the dataset:**
    - Download `YearPredictionMSD.txt` from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd) or [Kaggle](https://www.kaggle.com/datasets/uciml/millionsongdataset).
    - Place the downloaded `YearPredictionMSD.txt` file inside the `data/raw/` directory.

---

## Usage

1. **Process the Data:**  
   Run the data processing script first. From the `src` directory, execute:
    ```bash
    cd src
    python data_processing.py
    cd ..
    ```
    This will create the processed data files in `data/processed/`.

2. **Explore the Notebooks:**  
   The Jupyter notebooks in the `notebooks/` directory contain the main workflow and experiments. I recommend running them in order:
    - `1_Data_Exploration.ipynb`: Performs EDA on the raw data.
    - `2_Initial_Model_Runs.ipynb`: Compares baseline model architectures.
    - `3_Optimization_Experiments.ipynb`: Full optimization pipeline (LR finder, weight decay K-Fold, component tests, final training and evaluation).

    Start Jupyter Lab/Notebook from the project root:
    ```bash
    jupyter lab
    ```
    Then navigate to the `notebooks` folder and run the cells as needed.

---

## Technologies Used

- Python 3.x
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib & Seaborn
- Jupyter Notebook / Lab
- torch-lr-finder
- Jekyll & GitHub Pages (for the blog post)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## More Information

- **Blog Post:** [Building an Optimized Deep Neural Network: Lessons from Music Classification](https://mohitbhimrajka.com/blog/music_decade_classification)
- **GitHub Repository:** [https://github.com/MohitBhimrajka/music_decade_classification](https://github.com/MohitBhimrajka/music_decade_classification)

---

If you have questions or suggestions, feel free to open an issue or contact me.
