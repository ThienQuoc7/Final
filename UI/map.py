import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import plotly.express as px
import json
import plotly.graph_objects as go 
import plotly.express as px


# Function to load datasets
@st.cache_resource
def load_data():
    try:
        national_year = pd.read_csv("climdiv_national_year.csv")  # National data
        state_year = pd.read_csv("climdiv_state_year.csv")  # State data
        county_year = pd.read_csv("climdiv_county_year.csv")  # County data

        # Remove commas from 'year' and convert 'year' to integer type
        national_year['year'] = national_year['year'].astype(str).str.replace(',', '').astype(int)
        state_year['year'] = state_year['year'].astype(str).str.replace(',', '').astype(int)
        county_year['year'] = county_year['year'].astype(str).str.replace(',', '').astype(int)

        # Optionally convert 'year' to datetime if needed
        national_year['year'] = pd.to_datetime(national_year['year'], format='%Y')
        state_year['year'] = pd.to_datetime(state_year['year'], format='%Y')
        county_year['year'] = pd.to_datetime(county_year['year'], format='%Y')
        
        # Format FIPS codes
        state_year['fips'] = state_year['fips'].astype(str).str.zfill(2)  # 2 digits for states
        county_year['fips'] = county_year['fips'].astype(str).str.zfill(5)  # 5 digits for counties

        return national_year, state_year, county_year
    except Exception as e:
        st.error(f"Error loading CSV files: {e}")
        return None, None, None

# Function to load GeoJSON
@st.cache_resource
def load_county_geojson():
    try:
        with open("model_county.geojson", "r") as f:
            counties_geojson = json.load(f)
        return counties_geojson
    except Exception as e:
        st.error(f"Error loading GeoJSON file: {e}")
        return None

# Function to create sequences for LSTM model
def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length - 1):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Function to load the trained model and scaler
def load_trained_model_and_scaler(model_path, scaler_path):
    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception:
        return None, None

# Main function
def main():
    st.title("U.S. Temperature Analysis and Prediction")
    
    # Load datasets and GeoJSON
    national_year, state_year, county_year = load_data()
    counties_geojson = load_county_geojson()

    if national_year is None or state_year is None or county_year is None or counties_geojson is None:
        st.error("Error loading data, please check your files.")
        return

    # Sidebar options for navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Temperature Map", "Prediction with LSTM"])

    # Initialize session state variables
    if "selected_dataset" not in st.session_state:
        st.session_state.selected_dataset = None
    if "filtered_data" not in st.session_state:
        st.session_state.filtered_data = None

    if page == "Temperature Map":
        # Display map
        st.subheader("Interactive Temperature Map of the U.S. Counties (1895-2019)")

        # Sidebar for year selection
        year = st.sidebar.slider("Select Year", min_value=1895, max_value=2019, value=2019)

        # Filter data for the selected year
        county_data = county_year[county_year['year'].dt.year == year]

        # Ensure FIPS codes are strings with 5 digits
        county_data['fips'] = county_data['fips'].astype(str).str.zfill(5)

        # Extract FIPS codes from GeoJSON
        geo_fips = {feature['properties']['GEOID'] for feature in counties_geojson['features']}

        # Check for missing FIPS codes
        missing_fips = set(county_data['fips']) - geo_fips
        if missing_fips:
            st.warning(f"Missing FIPS codes in GeoJSON: {missing_fips}")

        # Fill missing temperature values with the national average
        national_avg_temp = national_year[national_year['year'].dt.year == year]['temp'].mean()
        county_data['temp'] = county_data['temp'].fillna(national_avg_temp)

        national_avg_temp_c = national_year[national_year['year'].dt.year == year]['tempc'].mean()
        county_data['tempc'] = county_data['tempc'].fillna(national_avg_temp_c)

        # Create the map
        fig = px.choropleth_mapbox(
            county_data,
            geojson=counties_geojson,
            locations='fips',
            color='tempc',
            color_continuous_scale="thermal",
            range_color=(county_data['tempc'].min(), county_data['tempc'].max()),
            mapbox_style="carto-positron",
            featureidkey="properties.GEOID",
            zoom=4,
            center={"lat": 37.8, "lon": -96},
            opacity=0.7,
            labels={'tempc': "Temperature (°C)"}
        )

        # Display map
        st.plotly_chart(fig, use_container_width=True)

        # Sidebar information
        st.sidebar.subheader("Selected Year Information")
        st.sidebar.write(f"Year: {year}")
        st.sidebar.write(f"Average Temperature (°F): {national_avg_temp:.2f}")
        st.sidebar.write(f"Average Temperature (°C): {national_avg_temp_c:.2f}")

    elif page == "Prediction with LSTM":
        # Display prediction section
        st.subheader("Temperature Prediction with LSTM Model")

        # Add buttons to choose dataset
        st.write("Choose dataset")
        if st.button("State"):
            st.session_state.selected_dataset = state_year
            st.write("Using State Data")
        if st.button("County"):
            st.session_state.selected_dataset = county_year
            st.write("Using County Data")
            
        # If dataset is selected, let the user pick FIPS code
        if st.session_state.selected_dataset is not None:
            data = st.session_state.selected_dataset
            fips_column = 'fips'
            counties = sorted(data[fips_column].unique())
            selected_fips = st.selectbox("Select a FIPS code", counties)

            # Filter data by selected FIPS and save to session state
            filtered_data = data[data[fips_column] == selected_fips]
            st.session_state.filtered_data = filtered_data

        # If filtered data is available, display temperature data
        if st.session_state.filtered_data is not None:
            filtered_data = st.session_state.filtered_data
            filtered_data_copy = filtered_data[['year', 'tempc']].copy()
            filtered_data_copy['year'] = pd.to_datetime(filtered_data_copy['year'], format='%Y')
            filtered_data_copy.set_index('year', inplace=True)
            st.subheader("Temperature Data")
            st.line_chart(filtered_data_copy['tempc'])

            # Train button
            if st.button('Predict Model'):
                st.write("Predicting the model...")

                # Prepare temperature data
                temp_data = filtered_data_copy[['tempc']].values

                # Scale the data
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(temp_data)

                # Prepare the data for LSTM
                sequence_length = 45  # Number of time steps for sequence
                X, Y = create_sequences(scaled_data, sequence_length)

                # Split the data
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

                # Reshape input for LSTM
                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                # File paths for saving models and scalers
                model_path = 'temperature_prediction_model.keras'
                scaler_path = 'scaler.pkl'

                # Train or load model
                model, scaler = load_trained_model_and_scaler(model_path, scaler_path)
                if model is None:
                    st.write("Training a new model...")
                    model = Sequential()
                    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
                    model.add(LSTM(units=50))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    model.fit(X_train, Y_train, epochs=100, batch_size=64, validation_split=0.1, verbose=1)
                    model.save(model_path)
                    joblib.dump(scaler, scaler_path)

                # Predictions
                predicted_temp = model.predict(X_test)
                predicted_temp = scaler.inverse_transform(predicted_temp)
                Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))

                # Evaluation metrics
                mse = mean_squared_error(Y_test, predicted_temp)
                mae = mean_absolute_error(Y_test, predicted_temp)
                rmse = mean_squared_error(Y_test, predicted_temp, squared=False)
                mape = mean_absolute_percentage_error(Y_test, predicted_temp)

                st.write(f"**Model Evaluation**")
                st.write(f"MSE: {mse}")
                st.write(f"MAE: {mae}")
                st.write(f'RMSE: {rmse}')
                st.write(f"MAPE: {mape}")
                
                # Plot prediction results with specific color for predicted temperature
                st.subheader("Predicted vs Actual Temperature")

                # Prepare DataFrame for actual and predicted temperatures
                results_df = pd.DataFrame({
                    'Actual Temperature': Y_test.flatten(),
                    'Predicted Temperature': predicted_temp.flatten()
                })

                # Create the Plotly figure
                fig = go.Figure()

                # Add actual temperature trace (default color)
                fig.add_trace(go.Scatter(
                    x=results_df.index,
                    y=results_df['Actual Temperature'],
                    mode='lines',
                    name='Actual Temperature', 
                ))

                # Add predicted temperature trace with coral color
                fig.add_trace(go.Scatter(
                    x=results_df.index,
                    y=results_df['Predicted Temperature'],
                    mode='lines',
                    name='Predicted Temperature',
                    line=dict(color='red')  # Coral color for predicted temperature line
                ))

                # Customize layout (optional, for better visual distinction)
                fig.update_layout(
                    xaxis_title="Data Points",
                    yaxis_title="Temperature (°C)",
                )

                # Show the plot
                st.plotly_chart(fig)

if __name__ == "__main__":
    main()
