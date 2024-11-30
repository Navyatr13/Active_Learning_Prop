# Active Learning for GNN

This repository implements an active learning pipeline to improve the performance of a custom graph neural network (GNN) using the QM9 dataset. The project also integrates a pretrained model for pseudo-labeling to guide the active learning process.

---

## **Table of Contents**
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Train and Test Custom GNN](#train-and-test-custom-gnn)
  - [Run Active Learning](#run-active-learning)
- [Project Structure](#project-structure)
- [Active Learning Process](#active-learning-process)
- [Results](#results)
- [Next Steps](#next-steps)

---

## **Project Overview**

The main goal of this project is to implement an **active learning loop** where:
1. A **custom GNN** is trained on labeled samples.
2. A **pretrained model** (DimeNet++) is used to pseudo-label the most uncertain data points from the unlabeled dataset.
3. The labeled dataset grows iteratively, improving the performance of the custom GNN.

### Dataset
The **QM9 dataset** is used, focusing on the `Atomization enthalpy at 298.15K` property.
---

## **Installation**

### Prerequisites
- Python 3.10+
- CUDA (if GPU is available)

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd entalpic-active-learning

2. Create and activate a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```

3. Install dependencies:
  ```bash
  pip install -r requirements.txt
  pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cu117.html
  ```

4. Test the installation:
  ```bash
  python main.py --batch_size 32 --active_learning
  ```

### Usage
### Train and Test Custom GNN
To train and test the custom GNN without active learning:
```
python main.py --batch_size 32
```
Run Active Learning
To start the active learning pipeline:
```
python main.py --batch_size 32 --num_cycles 5 --active_learning
```
Optional Arguments
--batch_size: Batch size for dataloaders (default: 32).
--num_batches: Number of batches for evaluation (default: 32).
--num_cycles: Number of active learning cycles (default: 5).
--active_learning: Activates the active learning loop.
