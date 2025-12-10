![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

# Mini Challenge: Virtual Sensor - AirLoadBench

## Project Description

This project was developed as part of a challenge to create a **virtual sensor** using machine learning methods. The main objective is to estimate the structural stress state at different points of an aircraft using only the parameters recorded by onboard instrumentation.

The proposed solution uses a supervised learning model to predict quantities that are not physically measured but inferred from flight variables such as attitude, speeds, accelerations, control commands, and wind conditions.

## Project Structure

The project structure consists of the following main components:

- **Data Preprocessing**:
  - Reading and concatenating `.csv` files containing flight data.
  - Creating new features and removing redundant variables.
  - Normalizing the data and splitting it into training, validation, and test sets.

- **Machine Learning Models**:
  - **MLP (Multilayer Perceptron)**: A classic neural network for static modeling.
  - **Multi-Model**: An architecture composed of an encoder, LSTM layers, and an MLP to capture spatial and temporal relationships.
  - **Optimized Multi-Model**: An optimized version of the multi-model with hyperparameters tuned using the Optuna library.

- **Evaluation**:
  - Metrics such as RMSE, MAE, and R² are used to evaluate model performance.
  - Comparison between different architectures and preprocessing strategies.

## Results

The best results were obtained with the **MLP model trained for 50 epochs**, achieving the following metrics:

- **RMSE**: 29.8818
- **MAE**: 19.9492
- **R²**: 0.9422

## Requirements

- Python 3.11
- Main libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `torch`
  - `optuna`

## How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Organize the data**:
   - Place the input `.csv` files in the `Data_AirLoadBench/` folder.

3. **Run the notebook**:
   - Open and execute the `Mini_Challenge_Bruno_Samuel_presentation.ipynb` file to train the models and evaluate the results.

4. **Testing**:
   - To evaluate the model on new data, use the test files in the `Data_AirLoadBench_test/` folder.

## Folder Structure

```
Mini_Challenge_Capteur_Virtuel_AirLoadBench/
├── Data_AirLoadBench/          # Training data
├── Data_AirLoadBench_test/     # Test data
├── Mini_Challenge_Bruno_Samuel_presentation.ipynb
├── README.md
└── requirements.txt
```

## Conclusion

This project demonstrated the effectiveness of machine learning models in predicting structural stresses in aircraft based on operational data. Despite the good results obtained, there is room for improvement, such as incorporating physics-informed neural networks (PINNs) and exploring more advanced architectures.

## Authors

- **Bruno Oliveira**
- **Samuel Ghezi**

Under the guidance of Professor **Martin Ghienne**.