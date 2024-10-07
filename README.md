![Screenshot](.streamlit/header.png "This is the header of the project")

# ds-air-pollution-prediction

## Executive Summary

This data science project aims to predict daily PM2.5 particulate matter concentrations across various locations in Africa. The dataset, originally provided for a **Zindi challenge in April 2020**, includes ground sensor, satellite, and weather data. Utilizing this data, we’ve developed multiple models that accurately forecast air quality, which is critical for public health and environmental monitoring. This report details the process from data collection to deploying a predictive application, providing insights and methodologies used throughout the project.

Our best model is a **Support Vector Regressor (SVR)**, providing a **Root Mean Square Error (RMSE)** of **22.91** on the test set. This performance surpasses the original winning solution's RMSE of **26.0997**, demonstrating that our model **outperforms the winning benchmark** from the Zindi competition by a significant margin. Additionally, models like **KNeighborsRegressor** and **ElasticNet** have achieved RMSE_test values of **23.34** and **23.59** respectively, further solidifying the robustness of our approach.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Results and Insights](#results-and-insights)
6. [Deployment](#deployment)
7. [Acknowledgments](#acknowledgments)
8. [License](#license)

## Introduction

The **ds-air-pollution-prediction** project leverages advanced analytics and machine learning to predict PM2.5 levels, aiding in the development of better environmental policies and health advisories. This project is essential for researchers, environmentalists, and policymakers engaged in air quality management.

## Project Structure

This project is organized into several Jupyter notebooks, testing scripts, and a Streamlit application that document each phase of the analytical process:

1. **01_data_collection.ipynb**: Data acquisition from various sources including Zindi and NOAA.
2. **02_data_preparation.ipynb**: Data cleaning and preprocessing for analysis readiness.
3. **03_exploratory_data_analysis.ipynb**: Exploratory analysis to uncover patterns and insights in the data.
4. **04_models.ipynb**: Model development and evaluation.
5. **05_project_summary_report.ipynb**: Compilation of findings, insights, and model performances.
6. **app.py**: Streamlit application for interactive visualization and exploration of model predictions.
7. **tests/**: Contains unit and integration tests to ensure data processing and model reliability.

### Version Control

The project utilizes **Git** for version control, ensuring that all changes are tracked and managed efficiently. The repository is hosted on GitHub, facilitating collaboration and version management.

### Requirements

All necessary dependencies are listed in the `requirements.txt` file, allowing for easy setup and replication of the project environment.

## Installation

To replicate and run this project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/johannesgooth/ds-air-pollution-prediction.git
   cd ds-air-pollution-prediction
   ```

2. **Set up a virtual environment (optional but recommended):**

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pyenv local 3.11.3
   python -m venv .mlflow_venv
   source .mlflow_venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt  
   ```

4. **Launch Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```

## Usage

Navigate through the project notebooks in order, starting from data collection to final model evaluation and app setup:

1. **Data Collection**: Execute `01_data_collection.ipynb`.
2. **Data Preparation**: Prepare the data with `02_data_preparation.ipynb`.
3. **Exploratory Data Analysis**: Analyze the data in `03_exploratory_data_analysis.ipynb`.
4. **Modeling**: Develop and evaluate models using `04_models.ipynb`.
5. **Summary and Reporting**: Review findings in `05_project_summary_report.ipynb`.
6. **Interactive Application**: Run `app.py` to explore model predictions through an interactive Streamlit app.

## Results and Insights

This project provides significant insights into the factors affecting air quality and develops robust models that surpass benchmarks, crucial for forecasting and managing the impacts of air pollution on public health. Key performance metrics from our models include:

- **Support Vector Regressor (SVR)**: RMSE_test = **22.91**, R²_test = **0.19**
- **KNeighborsRegressor**: RMSE_test = **23.34**, R²_test = **0.14**
- **ElasticNet**: RMSE_test = **23.59**, R²_test = **0.15**
- **Neural Networks (MLP)**: RMSE_test = **23.50**, R²_test = **0.15**
- **XGBRegressor**: RMSE_test = **23.99**, R²_test = **0.11**
- **AdaBoostRegressor**: RMSE_test = **24.09**, R²_test = **0.10**

These models collectively demonstrate the capability to make competitive and accurate predictions for air pollution levels, outperforming the original benchmark set by the Zindi competition.

## Deployment

We have developed the **PM2.5 Air Pollution Prediction App** using Streamlit, which allows users to interact with and visualize model predictions effectively. The app provides an intuitive interface where users can select specific IDs from the test dataset to view predicted versus actual PM2.5 values on the WHO Air Quality Index (AQI) scale. This interactive visualization aids in understanding model performance and the implications of air quality predictions.

### **About `app.py`**

The `app.py` script is the core of our Streamlit application. It performs the following functions:

- **Data Loading**: Reads and preprocesses the test data, including actual and predicted PM2.5 values.
- **Visualization**: Generates an AQI scale visualization with vertical lines representing predicted and actual PM2.5 levels for selected data points.
- **User Interaction**: Provides a sidebar for users to select any ID from the test dataset, dynamically updating the visualization and displaying key metrics like Actual PM2.5, Predicted PM2.5, and Absolute Error.
- **Error Handling**: Ensures smooth user experience by handling missing files and providing informative error messages.

To launch the app, run the following command in your terminal:

```bash
streamlit run app.py
```

This will start the Streamlit server and open the app in your default web browser.

## Testing

Comprehensive testing has been implemented to ensure the reliability and accuracy of data processing and model predictions. The `tests/` directory contains unit and integration tests that validate the functionality of various components within the project. These tests help in maintaining code quality and facilitate future enhancements.

To run the tests, navigate to the project directory and execute:

```bash
pytest tests/
```

Ensure that you have `pytest` installed, which can be added to your `requirements.txt` or installed separately:

```bash
pip install pytest
```

## Acknowledgments

Special thanks to the Zindi platform for providing the data and the challenge framework. Gratitude is also extended to NeueFische GmbH and to the contributors and the open-source community for their invaluable tools and resources, which made this project possible. Additionally, appreciation goes to the maintainers of Git for facilitating seamless version control and collaboration.

## License

This project is released under the MIT License.