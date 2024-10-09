import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
import io

# Function to load and preprocess data
def load_and_preprocess_data(file):
    df = pd.read_csv(file)
    df['Time Stamp'] = pd.to_datetime(df['Time Stamp (local standard time) yyyy-mm-ddThh:mm:ss'])
    df['Hour'] = df['Time Stamp'].dt.hour
    df['Month'] = df['Time Stamp'].dt.month
    df['DayOfYear'] = df['Time Stamp'].dt.dayofyear
    return df

# Function to perform classification
def classify_solar_conditions(df):
    features = [col for col in df.columns if col not in ['Time Stamp (local standard time) yyyy-mm-ddThh:mm:ss', 'Time Stamp']]
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    cluster_centers = kmeans.cluster_centers_
    irradiance_index = features.index('POA irradiance CMP22 pyranometer (W/m2)')
    irradiance_centers = cluster_centers[:, irradiance_index]
    cluster_labels = ['Mostly Cloudy', 'Partially Cloudy', 'Partially Sunny', 'Mostly Sunny']
    cluster_dict = dict(zip(np.argsort(irradiance_centers), cluster_labels))
    
    df['Predicted_Class'] = df['Cluster'].map(cluster_dict)
    
    return df

# Streamlit app
st.title('Solar Power Plant Data Analysis')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = load_and_preprocess_data(uploaded_file)
    df = classify_solar_conditions(df)
    
    st.success('File processed and classification completed!')
    
    # Date selection with a more user-friendly interface
    min_date = df['Time Stamp'].dt.date.min()
    max_date = df['Time Stamp'].dt.date.max()
    selected_date = st.date_input('Select a date to visualize', 
                                  min_value=min_date, max_value=max_date, value=min_date)
    
    # Filter data for selected date
    df_selected = df[df['Time Stamp'].dt.date == selected_date]
    
    # Plot classification results
    st.subheader('Solar Condition Classification')
    fig, ax = plt.subplots(figsize=(14, 7))
    scatter = ax.scatter(df_selected['Time Stamp'], df_selected['POA irradiance CMP22 pyranometer (W/m2)'], 
                         c=pd.Categorical(df_selected['Predicted_Class']).codes, cmap='viridis', alpha=0.6)
    ax.set_xlabel('Time')
    ax.set_ylabel('Irradiance (W/m2)')
    ax.set_title(f'Solar Conditions on {selected_date}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Use actual class names in the legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=class_name, 
                                  markerfacecolor=plt.cm.viridis(i/3), markersize=10)
                       for i, class_name in enumerate(df_selected['Predicted_Class'].unique())]
    ax.legend(handles=legend_elements, title="Classes", loc="upper right")
    
    st.pyplot(fig)
    
    # Plot solar power output
    st.subheader('Solar Power Output vs Time')
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df_selected['Time Stamp'], df_selected['Pmp (W)'], label='Power Output')
    ax.set_xlabel('Time')
    ax.set_ylabel('Power Output (W)')
    ax.set_title(f'Solar Power Output on {selected_date}')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)
    
    # Additional options for power output graph
    st.subheader('Customize Power Output Graph')
    y_axis_option = st.selectbox('Select Y-axis data', ['Pmp (W)', 'Voc (V)', 'Isc (A)', 'Vmp (V)', 'Imp (A)'])
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df_selected['Time Stamp'], df_selected[y_axis_option], label=y_axis_option)
    ax.set_xlabel('Time')
    ax.set_ylabel(y_axis_option)
    ax.set_title(f'{y_axis_option} vs Time on {selected_date}')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display classified CSV data for the selected date
    st.subheader(f'Classified Data for {selected_date}')
    st.dataframe(df_selected[['Time Stamp', 'Predicted_Class', 'POA irradiance CMP22 pyranometer (W/m2)', 'Pmp (W)','Voc (V)', 'Isc (A)', 'Vmp (V)', 'Imp (A)']])
    
    # Option to download processed data for the selected date
    csv_selected = df_selected.to_csv(index=False)
    st.download_button(
        label=f"Download processed data for {selected_date} as CSV",
        data=csv_selected,
        file_name=f"processed_solar_data_{selected_date}.csv",
        mime="text/csv",
    )

else:
    st.info('Please upload a CSV file to begin analysis.')
