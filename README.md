<<<<<<< HEAD
# Temperature Change Analysis and Prediction using LSTM models.

I am **Duong Thien Quoc**, the undergraduated student of **Greenwich Vietnam FPT University**.

This is my Final Project - COMP1682, which uses deep learning to analyze and predict temperature trends based on NOAA's historical climate data and regional temperature records. The project implements various statistical methods, LSTM models, and an interactive, user-friendly GUI for data exploration, prediction, and visualization of climate trends and model performance.

---

## Acknowledgments 

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Resources](#resources)

---

## Introduction 
- Climate change, a complex and multifaceted issue, has emerged as one of the most significant challenges of the 21st century. Its far-reaching consequences, including rising global temperatures, extreme weather events, and sea-level rise, pose a serious threat to ecosystems, economies, and human societies worldwide. To address this pressing issue, a comprehensive understanding of climate trends and patterns is essential.
- This project focuses on a detailed analysis of temperature changes in the United States from 1895 to 2019, utilizing a rich dataset provided by the National Oceanic and Atmospheric Administration (NOAA). By examining this historical record, I aim to uncover significant trends and patterns that shed light on the extent and pace of global warming.

---

## Features

- **Explore and Pre-Process Data**:

    - Examine regional variations and identify correlations between temperature anomalies and extreme weather events.
    - Encode geographic information (e.g., FIPS codes for states and counties) for accurate mapping.
    - Prepare time-series data for machine learning by creating sequences for training LSTM models.

- **Deep Learning**:
    - Use LSTM (Long Short-Term Memory) neural networks to model and predict temperature trends.
    - Validate predictions against historical records to ensure model reliability.
    - Assess LSTM predictions using metrics like Mean Absolute Percentage Error (MAPE).
    - Compare predictions to statistical baselines to demonstrate model efficacy.

- **GUI**:
    - Interactive Tools:
        - Build a Streamlit-based GUI to allow users to interact with the NOAA dataset and analysis results.
        - Include features for users to:
        Select regions and time periods for exploration.
        Visualize temperature trends and generate predictions.
        Customize visualizations, such as maps and charts, for detailed insights.
    - User-Friendly Design:
        - Provide an intuitive interface that enables non-experts to explore climate data and understand the findings.
        - Integrate tools for creating downloadable reports summarizing key trends and analyses.

---

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
    - **Data Processing**: NumPy, Pandas
    - **Visualization**: Matplotlib, Seaborn
    - **Deep Learning**: Scikit-learn, TensorFlow, Keras
- **GUI Development**: Streamlit
- **Tools**:
    - Google Collab
    - Visual Studio Code

## Installation

1.  **Create a virtual environment**:

- **Using `venv`**:
  ```bash
  python -m venv env
  source env/bin/activate   # Linux/MacOS
  env\Scripts\activate      # Windows
  ```

2. **Clone the repository**:
   ```bash
   git https://github.com/ThienQuoc7/Final/tree/main 
   cd Final
   ```
3. **Install packages in requirements.txt**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

How to use the GUI:

1. Navigate to the map.py file:
   ```bash
   cd Final/UI
   ```
2. Launch the app:
   ```bash
   streamlit run map.py
   ```

## Resources

| Path                                                         | Description                                  |
| :----------------------------------------------------------- | :------------------------------------------- |
| [Final]()                                                    | Main folder.                                 |
| &boxv;&nbsp; &boxvr;&nbsp; [UI]()                            | Main source code folder.                     |
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; [Data]()            | Contains code for data files.                |
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; [UI]()              | Code for the graphical user interface.       |
| &boxv;&nbsp; &boxvr;&nbsp; [Data]()                          | Folder for storing datasets.                 |
| &boxv;&nbsp; &boxvr;&nbsp; [images]()                        | Folder for storing generated images.         |
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; [images_gui]()      | GUI-related image outputs.                   |
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; [images_signal]()   | Signal-related plots.                        |
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; [images_spectrum]() | Spectrum-related visualizations.             |
| &boxv;&nbsp; &boxvr;&nbsp; [results]()                       | Stores results of best 219 SQI signal files. |
| &boxv;&nbsp; &boxvr;&nbsp; [save_models]()                   | Folder for saved deep learning model.        |
=======
# Temperature Change Analysis and Prediction using LSTM models.

I am **Duong Thien Quoc**, the undergraduated student of **Greenwich Vietnam FPT University**.

This is my Final Project - COMP1682, which uses deep learning to analyze and predict temperature trends based on NOAA's historical climate data and regional temperature records. The project implements various statistical methods, LSTM models, and an interactive, user-friendly GUI for data exploration, prediction, and visualization of climate trends and model performance.

---

## Acknowledgments 

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Resources](#resources)

---

## Introduction 
- Climate change, a complex and multifaceted issue, has emerged as one of the most significant challenges of the 21st century. Its far-reaching consequences, including rising global temperatures, extreme weather events, and sea-level rise, pose a serious threat to ecosystems, economies, and human societies worldwide. To address this pressing issue, a comprehensive understanding of climate trends and patterns is essential.
- This project focuses on a detailed analysis of temperature changes in the United States from 1895 to 2019, utilizing a rich dataset provided by the National Oceanic and Atmospheric Administration (NOAA). By examining this historical record, I aim to uncover significant trends and patterns that shed light on the extent and pace of global warming.

---

## Features

- **Explore and Pre-Process Data**:

    - Examine regional variations and identify correlations between temperature anomalies and extreme weather events.
    - Encode geographic information (e.g., FIPS codes for states and counties) for accurate mapping.
    - Prepare time-series data for machine learning by creating sequences for training LSTM models.

- **Deep Learning**:
    - Use LSTM (Long Short-Term Memory) neural networks to model and predict temperature trends.
    - Validate predictions against historical records to ensure model reliability.
    - Assess LSTM predictions using metrics like Mean Absolute Percentage Error (MAPE).
    - Compare predictions to statistical baselines to demonstrate model efficacy.

- **GUI**:
    - Interactive Tools:
        - Build a Streamlit-based GUI to allow users to interact with the NOAA dataset and analysis results.
        - Include features for users to:
        Select regions and time periods for exploration.
        Visualize temperature trends and generate predictions.
        Customize visualizations, such as maps and charts, for detailed insights.
    - User-Friendly Design:
        - Provide an intuitive interface that enables non-experts to explore climate data and understand the findings.
        - Integrate tools for creating downloadable reports summarizing key trends and analyses.

---

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
    - **Data Processing**: NumPy, Pandas
    - **Visualization**: Matplotlib, Seaborn
    - **Deep Learning**: Scikit-learn, TensorFlow, Keras
- **GUI Development**: Streamlit
- **Tools**:
    - Google Collab
    - Visual Studio Code

## Installation

1.  **Create a virtual environment**:

- **Using `venv`**:
  ```bash
  python -m venv env
  source env/bin/activate   # Linux/MacOS
  env\Scripts\activate      # Windows
  ```

2. **Clone the repository**:
   ```bash
   git https://github.com/ThienQuoc7/Final/tree/main 
   cd Final
   ```
3. **Install packages in requirements.txt**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

How to use the GUI:

1. Navigate to the map.py file:
   ```bash
   cd Final/UI
   ```
2. Launch the app:
   ```bash
   streamlit run map.py
   ```

## Resources

| Path                                                         | Description                                  |
| :----------------------------------------------------------- | :------------------------------------------- |
| [Final]()                                                    | Main folder.                                 |
| &boxv;&nbsp; &boxvr;&nbsp; [UI]()                            | Main source code folder.                     |
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; [Data]()            | Contains code for data files.                |
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; [UI]()              | Code for the graphical user interface.       |
| &boxv;&nbsp; &boxvr;&nbsp; [Data]()                          | Folder for storing datasets.                 |
| &boxv;&nbsp; &boxvr;&nbsp; [images]()                        | Folder for storing generated images.         |
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; [images_gui]()      | GUI-related image outputs.                   |
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; [images_signal]()   | Signal-related plots.                        |
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; [images_spectrum]() | Spectrum-related visualizations.             |
| &boxv;&nbsp; &boxvr;&nbsp; [results]()                       | Stores results of best 219 SQI signal files. |
| &boxv;&nbsp; &boxvr;&nbsp; [save_models]()                   | Folder for saved deep learning model.        |
>>>>>>> 649f0b26bce32c4a9845c9520e00db4aee502ebd
