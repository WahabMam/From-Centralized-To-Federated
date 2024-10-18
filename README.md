# From Centralized to Federated Learning

## Project Overview
This repository demonstrates the migration from centralized to federated learning using a simple model with the MNIST dataset and Pytorch framework. It serves as a beginner-friendly demo, showcasing how to transition from a centralized system to federated learning while highlighting the advantages of this approach. We have created notebooks that offer comprehensive and clear instructions before each cell of code, making it easy for newcomers to follow along and understand the concepts involved
## Getting Started

### Prerequisites
- **Python**: Ensure you have Python version 3.9.6 installed. 

### Creating a Virtual Environment
1. **Create a virtual environment**:
   ```bash
   python -m venv venv --prompt myenv  # Replace 'myenv' with your preferred environment name
   ```

2. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

### Installation
3. **Install required libraries**: Create a `requirements.txt` file with the necessary packages (e.g., `tensorflow`, `flwr`) and include the specific versions. Then run:
   ```bash
   pip install -r requirements.txt
   ```

### Utility Functions
4. **Review utility files**: Take a look at `Utils1.py` and `Utils2.py`, which contain utility functions that are helpful throughout the notebooks.

### Running the Notebooks
5. **Open and run the notebooks in the following order for better understanding**:
   - `Centralized_Notebook.ipynb`: Understand the centralized training approach.
   - `Federated_Notebook.ipynb`: Explore the federated learning setup.
   - `Comparison.ipynb`: Compare the results and insights from both approaches.
 