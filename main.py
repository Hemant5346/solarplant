import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
import io
import datetime

# Set page configuration
st.set_page_config(page_title="Solar Power Analysis", layout="wide")

# Custom CSS to improve UI
st.markdown("""
<style>
    .stButton>button {
        color: #4F8BF9;
        border-radius: 50px;
        height: 3em;
        width: 100%;
    }
    .stDateInput>div>div>input {
        color: #4F8BF9;
    }
    .stSelectbox>div>div>select {
        color: #4F8BF9;
    }
    .css-1d391kg {
        padding-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

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

# Function to format time for x-axis
def format_time(x, _):
    hours, remainder = divmod(int(x), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}"

# Streamlit app
st.title('☀️ Solar Power Plant Data Analysis')

# Sidebar for file upload and date selection
with st.sidebar:
    st.header("📊 Data Input")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = load_and_preprocess_data(uploaded_file)
    df = classify_solar_conditions(df)
    
    st.sidebar.success('🎉 File processed and classification completed!')
    
    # Date selection in sidebar
    st.sidebar.header("📅 Date Selection")
    min_date = df['Time Stamp'].dt.date.min()
    max_date = df['Time Stamp'].dt.date.max()
    selected_date = st.sidebar.date_input('Select a date to visualize', 
                                          min_value=min_date, max_value=max_date, value=min_date)
    
    # Filter data for selected date
    df_selected = df[df['Time Stamp'].dt.date == selected_date]
    
    # Display summary statistics
    st.header("📊 Summary Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Irradiance", f"{df_selected['POA irradiance CMP22 pyranometer (W/m2)'].mean():.2f} W/m²")
    col2.metric("Peak Power Output", f"{df_selected['Pmp (W)'].max():.2f} W")
    col3.metric("Predominant Condition", df_selected['Predicted_Class'].mode().values[0])
    
    # Plot classification results
    st.header('🌤️ Solar Condition Classification')
    fig, ax = plt.subplots(figsize=(14, 7))

    time_in_seconds = df_selected['Time Stamp'].dt.hour * 3600 + df_selected['Time Stamp'].dt.minute * 60 + df_selected['Time Stamp'].dt.second

    # Create a color map
    unique_classes = df_selected['Predicted_Class'].unique()
    color_map = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(unique_classes)))
    color_dict = dict(zip(unique_classes, color_map))

    # Plot each class separately
    for class_name in unique_classes:
        mask = df_selected['Predicted_Class'] == class_name
        ax.scatter(time_in_seconds[mask], 
                df_selected.loc[mask, 'POA irradiance CMP22 pyranometer (W/m2)'], 
                c=[color_dict[class_name]], 
                label=class_name, 
                alpha=0.6)

    ax.set_xlabel('Time')
    ax.set_ylabel('Irradiance (W/m2)')
    ax.set_title(f'Solar Conditions on {selected_date}')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_time))
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Add legend
    ax.legend(title="Classes", loc="upper right")

    st.pyplot(fig)

    
    # Plot solar power output
    st.header('⚡ Solar Power Output Analysis')
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(time_in_seconds, df_selected['Pmp (W)'], label='Power Output')
    ax.set_xlabel('Time')
    ax.set_ylabel('Power Output (W)')
    ax.set_title(f'Solar Power Output on {selected_date}')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_time))
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)
    
    # Additional options for power output graph
    st.header('🔧 Customize Power Output Graph')
    y_axis_option = st.selectbox('Select Y-axis data', ['Pmp (W)', 'Voc (V)', 'Isc (A)', 'Vmp (V)', 'Imp (A)'])
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(time_in_seconds, df_selected[y_axis_option], label=y_axis_option)
    ax.set_xlabel('Time')
    ax.set_ylabel(y_axis_option)
    ax.set_title(f'{y_axis_option} vs Time on {selected_date}')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_time))
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display classified CSV data for the selected date
    st.header(f'📅 Classified Data for {selected_date}')
    st.dataframe(df_selected[['Time Stamp', 'Predicted_Class', 'POA irradiance CMP22 pyranometer (W/m2)', 'Pmp (W)','Voc (V)', 'Isc (A)', 'Vmp (V)', 'Imp (A)']])
    
    # Option to download processed data for the selected date
    csv_selected = df_selected.to_csv(index=False)
    st.download_button(
        label=f"📥 Download processed data for {selected_date} as CSV",
        data=csv_selected,
        file_name=f"processed_solar_data_{selected_date}.csv",
        mime="text/csv",
    )

else:
    st.info('👆 Please upload a CSV file in the sidebar to begin analysis.')

# Footer
st.markdown("---")
st.markdown("Created by Hemant")
