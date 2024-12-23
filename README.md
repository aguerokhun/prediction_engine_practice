# Planting-time Prediction Engine 🌱

This project is a machine learning-powered prediction engine that assists farmers in identifying the best planting seasons based on their crop type and location. The engine is fine-tuned with custom endpoints to demonstrate its functionality for testing purposes.

## 🌟 Features
- **Regression Model**:
  - Predicts numerical planting metrics, such as the best planting month or expected yield, based on crop and location inputs.

- **Classification Model**:
  - Predicts planting feasibility by classifying crops into discrete categories, such as:
    - **"Suitable"**: Indicates optimal planting conditions.
    - **"Not Suitable"**: Suggests suboptimal conditions for planting.
  - Inputs include crop type (`label`), country, and processed numerical data from the regression model.
  - Trained using `Crop_Data.csv`, which contains labeled data linking crops to planting outcomes.

- **Endpoints for Testing**:
  - Accessible endpoints for testing and understanding the engine’s functionality.

- **Visualization**:
  - Includes visual representations of residuals for performance evaluation.

## 🚀 How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/agueroKhun/prediction_engine_practice.git

## Install the required dependencies:
pip install -r requirements.txt

## Run the script to test the endpoints:
python prediction_engine_practice.py

## Input data:
Provide crop type (label) and country.
The regression model predicts numerical planting values, which are passed to the classification model for feasibility classification.

## 📊 Dataset Information
The dataset used is limited and may not generalize to all scenarios.
Files:
Crop_Data.csv: Contains crop-specific training data.
metrics_results.csv: Logs model evaluation metrics.

## 🛠️ Tech Stack
Programming Language: Python
Libraries: Scikit-Learn, Pandas, NumPy
Models: Pre-trained regression and classification models (.pkl files)

## 📈 Visualizations
Residual histograms provide insights into the model's prediction errors.
Visualizations are available in the residuals_histograms/ folder.

## 🔗 Future Enhancements
Extend the dataset to include more diverse crops and regions.
Improve model accuracy with additional features like soil quality and weather data.
Integrate a user-friendly dashboard for easier interaction with the engine.

## ❗ Notes
This project is for demonstration and learning purposes.
The current implementation is designed for testing with limited data.
If you'd like to contribute or explore more, feel free to contact me!

## 📫 Contact
Email: sanusitaiwo10@gmail.com
