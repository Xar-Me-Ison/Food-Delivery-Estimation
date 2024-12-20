# Food Delivery Time Estimation

## Overview

The **Food Delivery Time Estimation** project utilizes machine learning models to predict the delivery time of food orders based on various factors such as geographical distance, delivery driver attributes, and vehicle type. This project leverages Python and libraries like Pandas, Scikit-learn, and Streamlit to provide a user-friendly web interface for predictions and data analysis.

## Features

- **Interactive Web Application**: Built with Streamlit to provide a seamless user interface.
- **Multiple ML Algorithms**: Supports models such as Random Forest, Gradient Boosting, Linear Regression, and more.
- **Data Visualization**: Includes detailed exploratory data analysis with visualizations using Matplotlib and Seaborn.
- **Distance Calculation**: Implements the Haversine formula for precise distance computation between locations.
- **Customizable Inputs**: Allows users to input data dynamically for real-time predictions.

## Technologies Used

- **Programming Language**: Python
- **Data Processing**: Pandas, Numpy
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Web Interface**: Streamlit
- **Big Data Processing**: PySpark (optional for larger datasets)

## Dataset

The project uses a food delivery dataset sourced from Kaggle, which contains details such as:

- Restaurant and delivery location coordinates (latitude and longitude).
- Delivery driver attributes (e.g., age, vehicle type).
- Delivery time taken (in minutes).

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/food-delivery-time-estimation.git
   ```
2. Navigate to the project directory:
   ```bash
   cd food-delivery-time-estimation
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```
## Usage

1. Open the application in your browser.

2. Navigate between the following tabs:
    - Application: Input details to estimate delivery time.
    - Data Analysis: Explore and visualize the dataset.
    - README: View project documentation directly within the app.

3. Choose a machine learning algorithm and provide necessary inputs like restaurant and customer coordinates, vehicle type, and driver age.

4. View estimated delivery time and additional model accuracy metrics.

## Project Workflow

### 1. Data Processing
- Clean and preprocess the dataset.
- Compute the distance between restaurant and delivery location using the Haversine formula.
- Convert categorical variables (e.g., vehicle type) to numerical using one-hot encoding.

### 2. Model Training
- Train multiple machine learning models using Scikit-learn.
- Evaluate models based on metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared.

### 3. Real-Time Predictions
- Accept user inputs via the Streamlit interface.
- Perform predictions using the selected machine learning model.
- Display results along with optional performance metrics.

## Exploratory Data Analysis (EDA)
The project includes detailed EDA with:
- Distribution analysis of key variables.
- Correlation heatmaps.
- Summary statistics for numeric data.

### Correlational Analysis
In which, we calculated the correlation between each numerical column and 'Time_taken(min)` column to identify the relevant variables that impact the delivery time.
![image](https://github.com/user-attachments/assets/5fe2942b-9f6c-4d1a-80e1-cc45baea6b65)

### Geospatial Visualisation 
A scatter plot was created to represent the geographic locations of restaurants and delivery points using latitude and longitude. Using the defined Haversine method, a new column 'distance' was added to the Data Frame for further analysis.

![image](https://github.com/user-attachments/assets/18243ff5-3987-4bdd-9fba-f74fd822a861)

#### Haversine Method
```python
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Το ρ της γης είναι 6371 km.


    # Μετατροπή από γεωγραφικού πλάτους και μήκους σε radians.
    lat1 = mt.radians(lat1)
    lon1 = mt.radians(lon1)
    lat2 = mt.radians(lat2)
    lon2 = mt.radians(lon2)


    # Υπολογισμός διαφερός κάθε γεωγραφικής θέσεις
    Δlat = lat2 - lat1
    Δlon = lon2 - lon1


    # Φόρμουλα Haversine
    a = mt.sin(Δlat/2)**2 + mt.cos(lat1) * mt.cos(lat2) * mt.sin(Δlon/2)**2
    c = 2 * mt.atan2(mt.sqrt(a), mt.sqrt(1-a))
    d = R * c


    return d

```

### Temporal Analysis
By analysing the distribution of delivery times based on the type of vehicle, we ploted the kernel density estimation (KDE) plot.
![image](https://github.com/user-attachments/assets/7bef09bb-3cdb-42d6-8527-7878d4bce670)

### Age Distribution
To visualise the age distribution of delivery drivers using a histogram.
![image](https://github.com/user-attachments/assets/3bc1e604-4d2f-4a92-894e-0c56acd82630)


## Results
The models were evaluated based on:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)

Random Forest Regression provided the best results with the highest R² score and lowest error metrics.

## Streamlit Web Interface
![image](https://github.com/user-attachments/assets/a98c46b3-d1ff-4ea9-960b-432c63c754eb)
![image](https://github.com/user-attachments/assets/be19b4c7-8f9f-483b-9e2f-5234c0833a40)

The vast array of machine learning models such as: Random Forest Regression, Gradient Boosting Regression, Linear Regression, Decision Tree Regression, Extra Trees Fores and K-Neighbors Regression. 

![image](https://github.com/user-attachments/assets/855e16b4-fd71-4b52-be1f-9a8b67a2eeff)

Furthermore, within the confines of the Web Interface the user can see the Exploratory Data Analysis and analyse the data, as we did, step by step. Aiding in the knowledge and total understanding of Data Analysis in general.

![image](https://github.com/user-attachments/assets/a5232828-268b-47d4-b4c1-f74691379e2f)
![image](https://github.com/user-attachments/assets/ce10b2cb-ae11-4f56-b2df-41e3a099678a)
![image](https://github.com/user-attachments/assets/986d55a2-8fab-4c53-b8f3-3b665a97f071)

Include screenshots here to showcase the Streamlit application, demonstrating features such as:
Input forms for predictions.
Model selection options.
Prediction results and data visualization tabs.

## Future Enhancements
- Incorporate additional features like weather and traffic data.
- Support larger datasets using distributed computing with PySpark.
- Optimize models for better performance.

## Contributors
- Vasileios Katotomichelakis (Π2020132)
- Charalampos Makrylakis (Π2019214)

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
Special thanks to Kaggle for the dataset and open-source libraries used in this project.
