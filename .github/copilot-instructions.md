# AI Agent Coding Guidelines for LLM Classifier

## Project Overview
This project is an LLM-based classifier with the following key components:
- **Feature Extraction**: Uses DeBERTa for generating features from raw data.
- **Model Training**: Implements a LightGBM training framework.
- **Hyperparameter Optimization (HPO)**: Includes a baseline Random Search algorithm and supports custom HPO algorithms.

### Key Directories and Files
- `src/`
  - `feature_extraction.py`: Handles feature extraction.
  - `train.py`: Manages model training.
  - `hpo/`: Contains HPO-related code.
    - `base_hpo.py`: Base class for HPO algorithms.
    - `random_search.py`: Example implementation of Random Search.
- `data/`
  - `raw/`: Stores raw input data.
  - `processed/`: Contains processed features for training and testing.
- `outputs/`: Stores model outputs and submission files.
- `models/`: Saves trained models and HPO history.

## Developer Workflows

### Environment Setup
1. Create and activate the Python environment:
   ```bash
   conda create -n llm_classifier python=3.9
   conda activate llm_classifier
   pip install -r requirements.txt
   ```

### Running the Project
- **Default HPO with Random Search**:
  ```bash
  python main.py
  ```
- **Feature Extraction**:
  ```bash
  python main.py --mode extract
  ```
- **Custom HPO Algorithm**:
  ```bash
  python main.py --mode hpo --algo your_algo
  ```

### Adding a New HPO Algorithm
1. Create a new file in `src/hpo/` (e.g., `your_algo.py`).
2. Define your algorithm by inheriting `BaseHPO`.
3. Register the algorithm in `src/hpo/__init__.py`:
   ```python
   from .your_algo import YourAlgorithm

   AVAILABLE_ALGORITHMS = {
       'random': RandomSearch,
       'your_algo': YourAlgorithm,  # Add your algorithm here
   }
   ```
4. Run the algorithm:
   ```bash
   python main.py --mode hpo --algo your_algo
   ```

## Project-Specific Conventions
- **Search Space Definition**: Each HPO algorithm defines its own search space in its implementation. This allows flexibility for different optimization techniques.
- **Minimal Configuration**: The project avoids external configuration files. All settings are defined in the code.
- **Output Structure**:
  - `outputs/`: Stores submission files.
  - `models/`: Contains trained models and HPO history.

## Integration Points
- **Dependencies**: Ensure `requirements.txt` is up-to-date when adding new libraries.
- **Cross-Component Communication**: Processed features in `data/processed/` are shared between feature extraction and training components.

## Examples
- **Check Available Algorithms**:
  ```bash
  python main.py --algo unknown
  ```
- **Run with Bayesian Optimization**:
  ```bash
  python main.py --algo bayesian
  ```

## Common Issues
- **Missing Features**: Run feature extraction first:
  ```bash
  python main.py --mode extract
  ```
- **Unknown Algorithm**: Ensure the algorithm is registered in `src/hpo/__init__.py`.

---

This document is a living guide. Update it as the project evolves.