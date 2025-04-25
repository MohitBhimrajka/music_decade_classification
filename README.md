# Music Decade Classification

This project aims to classify music into different decades using machine learning techniques. The project uses the Million Song Dataset (MSD) to train and evaluate models for music decade classification.

## Project Structure

```
music_decade_classification/
├── data/               # Data directory
│   ├── raw/           # Raw data files
│   └── processed/     # Processed data files
├── src/               # Source code
├── notebooks/         # Jupyter notebooks
├── results/           # Results and outputs
├── reports/           # Project reports
└── blog/              # Blog post drafts
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
- Place the YearPredictionMSD.txt file in the `data/raw/` directory

## Usage

1. Data Processing:
```bash
python src/data_processing.py
```

2. Training:
```bash
python src/train.py
```

3. For interactive exploration, use the Jupyter notebooks in the `notebooks/` directory.

## Project Components

- `data_processing.py`: Handles data loading, cleaning, and preprocessing
- `models.py`: Contains neural network architectures
- `train.py`: Implements training and evaluation logic
- `lr_finder.py`: Learning rate finder implementation
- `optimize.py`: Scripts for model optimization
- `utils.py`: Helper functions and utilities

## License

This project is licensed under the MIT License - see the LICENSE file for details. 