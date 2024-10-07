import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
from PIL import Image

# Function to load and concatenate data based on index
@st.cache_data
def load_data():
    # Load the CSV files
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')
    y_pred = pd.read_csv('data/y_test_pred.csv')
    
    # Strip any leading/trailing spaces in column names
    X_test = X_test.rename(columns=lambda x: x.strip())
    y_test = y_test.rename(columns=lambda x: x.strip())
    y_pred = y_pred.rename(columns=lambda x: x.strip())
    
    # Rename columns to standard names
    if 'pm2_5' in y_test.columns:
        y_test = y_test.rename(columns={'pm2_5': 'pm2.5_actual'})
    else:
        st.error("Column 'pm2_5' not found in y_test.csv.")
        st.stop()
        
    if 'PM2.5_Prediction' in y_pred.columns:
        y_pred = y_pred.rename(columns={'PM2.5_Prediction': 'pm2.5_pred'})
    else:
        st.error("Column 'PM2.5_Prediction' not found in y_test_pred.csv.")
        st.stop()
    
    # Reset index to ensure proper alignment
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)
    
    # Concatenate the DataFrames horizontally
    data = pd.concat([X_test, y_test, y_pred], axis=1)
    
    return data

# Function to calculate overall RMSE
def calculate_rmse(actual, predicted):
    return math.sqrt(mean_squared_error(actual, predicted))

# Function to plot AQI scale with vertical lines for Predicted and Actual PM2.5
def plot_aqi_scale(pm25_pred, pm25_actual, rmse, font_size=10, line_height=1.0, line_width=2, legend_x=0.9):
    """
    Plots the WHO AQI scale with vertical lines for Predicted and Actual PM2.5 values.

    Parameters:
    - pm25_pred (float): Predicted PM2.5 value.
    - pm25_actual (float): Actual PM2.5 value.
    - rmse (float): Root Mean Squared Error of the model.
    - font_size (int): Font size for category labels and legend.
    - line_height (float): Total height of the vertical lines.
    - line_width (int): Thickness of the vertical lines.
    - legend_x (float): Horizontal position of the legend (0 to 1 scale).

    Returns:
    - fig (matplotlib.figure.Figure): The generated AQI scale figure.
    """
    # Define updated AQI categories for PM2.5
    categories = [
        {"label": "Excellent", "min": 0, "max": 12, "color": "#223aab"},
        {"label": "Good", "min": 12.1, "max": 35.4, "color": "#b9d5e1"},
        {"label": "Moderate", "min": 35.5, "max": 55.4, "color": "#628b2d"},
        {"label": "Poor", "min": 55.5, "max": 150.4, "color": "#ffbc06"},
        {"label": "Very Poor", "min": 150.5, "max": 250.4, "color": "#fb8db8"},
        {"label": "Severe", "min": 250.5, "max": 500, "color": "#d63331"},
    ]

    fig, ax = plt.subplots(figsize=(14, 4))  # Increased height for better line visibility

    # Plot each category as a horizontal bar with a semi-transparent background
    for category in categories:
        ax.barh(0, category['max'] - category['min'], left=category['min'],
                color=category['color'], edgecolor='black', alpha=0.6, height=1.0)

    # Remove y-axis
    ax.get_yaxis().set_visible(False)
    for spine in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine].set_visible(False)

    # Set x-axis limits based on the maximum category
    ax.set_xlim(0, max(cat['max'] for cat in categories) + 50)

    # Add rotated labels for categories inside their respective boxes
    for category in categories:
        ax.text((category['min'] + category['max']) / 2, 0, category['label'],
                horizontalalignment='center', verticalalignment='center', fontsize=font_size, 
                rotation=90, color='white', fontweight='bold')

    # Define the vertical line parameters
    y_bottom = -line_height / 2
    y_top = line_height / 2

    # Plot the Predicted PM2.5 value as a solid black vertical line
    ax.plot([pm25_pred, pm25_pred], [y_bottom, y_top], color='black', linewidth=line_width, label='Predicted PM2.5')

    # Plot the Actual PM2.5 value as a dashed black vertical line
    ax.plot([pm25_actual, pm25_actual], [y_bottom, y_top], color='black', linewidth=line_width, linestyle=':', label='Actual PM2.5')

    # Add a legend with fully transparent frame and background, and adjustable horizontal position
    legend = ax.legend(bbox_to_anchor=(legend_x, 0.5), loc='center', fontsize=font_size, frameon=False)
    legend.get_frame().set_alpha(0)  # Set legend background to fully transparent

    plt.tight_layout()
    return fig

# Main Streamlit App
def main():
    st.set_page_config(page_title="PM2.5 Prediction Visualization", layout="wide")

    # Set the manually tunable image size for the top image
    image_width = 600  # You can manually adjust this value to change the size of the image
    try:
        image = Image.open('.streamlit/app_header.png')
        st.image(image, width=image_width)
    except FileNotFoundError:
        st.error("Header image not found. Please ensure 'app_header.png' is in the '.streamlit/' directory.")

    # Load data before sidebar interaction
    data = load_data()

    # Calculate overall RMSE
    overall_rmse = calculate_rmse(data['pm2.5_actual'], data['pm2.5_pred'])

    # Calculate absolute error for each data point
    data['absolute_error'] = abs(data['pm2.5_actual'] - data['pm2.5_pred'])

    # Sidebar configuration
    try:
        sidebar_image = Image.open('.streamlit/app_sidebar_header.png')
        st.sidebar.image(sidebar_image, use_column_width=True)
    except FileNotFoundError:
        st.sidebar.error("Sidebar header image not found. Please ensure 'app_sidebar_header.png' is in the '.streamlit/' directory.")

    # Sidebar for ID selection
    st.sidebar.header("Select ID")

    # Select box to choose from all IDs
    selected_id = st.sidebar.selectbox(" ", data['id'].unique())

    # Display selected ID next to the AQI text
    st.write(f"Select an ID to view the predicted PM2.5 value on the WHO Air Quality Index (AQI) scale.")

    # Main app content

    # Get data for selected ID
    selected_data = data[data['id'] == selected_id].iloc[0]
    pm25_pred = round(selected_data['pm2.5_pred'], 1)
    pm25_actual = round(selected_data['pm2.5_actual'], 1)

    # Calculate absolute error for the selected datapoint
    absolute_error = round(abs(pm25_actual - pm25_pred), 1)

    # Customize line height, width, and legend position
    line_height = 1.0  # Increased line height for longer lines
    line_width = 2      # Increased line width for better visibility
    legend_x = 1.05     # Customize horizontal position of the legend here

    # Plot AQI scale with vertical lines for Predicted and Actual PM2.5
    fig = plot_aqi_scale(pm25_pred, pm25_actual, overall_rmse, font_size=18, line_height=line_height, line_width=line_width, legend_x=legend_x)

    # Display the AQI scale directly under the "Select an ID..." text
    st.pyplot(fig)

    # Display Actual, Predicted, and Error values in a row below the AQI scale
    col1, col2, col3 = st.columns(3)
    col1.metric("Actual PM2.5", f"{pm25_actual} μg/m³")
    col2.metric("Predicted PM2.5", f"{pm25_pred} μg/m³")
    col3.metric("Absolute Error", f"{absolute_error} μg/m³")

if __name__ == "__main__":
    main()