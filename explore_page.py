import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile

# Load data
st.cache
def load_data():
    # Unzip the file
    with zipfile.ZipFile("game_csv.zip", "y") as zip_ref:
        zip_ref.extractall("game_data")
    # Check if the folder exists
    if os.path.exists("survey_results_public"):
        # Access the folder and load the CSV file into a DataFrame
        csv_file_path = os.path.join("game_data", "game.csv")
        # Check if the folder
        #folder_path = '/Users/wallyk./Desktop/CB-DS-17/Kaddle Basketball & Final Project/csv'
        data = pd.read_csv(csv_file_path)
    return data

def show_explore_page():
    data = load_data()

    # Data cleaning
    threshold = 0.5
    recent_data = data.loc[:, data.isnull().mean() < threshold]
    recent_data = recent_data.dropna()
    
    # Ensure wl_home is in the data and map correctly
    if 'wl_home' in recent_data.columns:
        recent_data['wl_home'] = recent_data['wl_home'].map({'W': 1, 'L': 0})
    else:
        st.error("'wl_home' column is missing from the data.")
        return
    
    numeric_cols = recent_data.select_dtypes(include=[np.number])
    means = numeric_cols.mean()
    recent_data.fillna(means, inplace=True)

    # Streamlit app
    st.title("Explore Basketball Stats")
    st.write("### Visualization from 1980 to 2023")

    st.write("## Data Information")
    st.write(data.info())
    st.write(data.head())

    st.write("## Cleaned Data Summary")
    st.write(recent_data.describe())

    # Visualization
    st.write("## Distribution of Points Scored by Home Team")
    fig, ax = plt.subplots()
    sns.histplot(recent_data['pts_home'], bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of Points Scored by Home Team')
    ax.set_xlabel('Points')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Select key columns for histograms
    offensive_metrics = [
        'pts_home', 'pts_away', 'fgm_home', 'fgm_away',
        'fga_home', 'fga_away', 'fg_pct_home', 'fg_pct_away',
        'fg3m_home', 'fg3m_away', 'ftm_home', 'ftm_away',
        'fta_home', 'fta_away', 'ft_pct_home', 'ft_pct_away',
        'ast_home', 'ast_away', 'oreb_home', 'oreb_away'
    ]

    # Calculate the difference between offensive and defensive rebounds
    recent_data['oreb_diff_home'] = recent_data['dreb_home'] - recent_data['oreb_home']
    recent_data['oreb_diff_away'] = recent_data['dreb_away'] - recent_data['oreb_away']

    # Add these new columns to the respective metrics lists
    offensive_metrics.extend(['oreb_diff_home', 'oreb_diff_away'])

    # Visualize offensive metrics
    st.write("## Offensive Metrics")
    fig, axes = plt.subplots(8, 3, figsize=(15, 24))
    axes = axes.flatten()
    for i, col in enumerate(offensive_metrics):
        sns.histplot(recent_data[col], bins=15, kde=True, color='skyblue', ax=axes[i])
        axes[i].set_title(col)
    plt.tight_layout()
    st.pyplot(fig)

    # Defensive metrics
    defensive_metrics = [
        'reb_home', 'reb_away', 'tov_home', 'tov_away',
        'pf_home', 'pf_away', 'blk_home', 'blk_away',
        'stl_home', 'stl_away', 'dreb_home', 'dreb_away'
    ]

    # Add the new columns to the respective metrics lists
    defensive_metrics.extend(['oreb_diff_home', 'oreb_diff_away'])

    # Visualize defensive metrics
    st.write("## Defensive Metrics")
    fig, axes = plt.subplots(6, 3, figsize=(15, 18))
    axes = axes.flatten()
    for i, col in enumerate(defensive_metrics):
        sns.histplot(recent_data[col], bins=15, kde=True, color='lightgreen', ax=axes[i])
        axes[i].set_title(col)
    plt.tight_layout()
    st.pyplot(fig)

    # Play stats
    play_stats = ['plus_minus_home', 'plus_minus_away']

    # Visualize additional play statistics
    st.write("## Play Statistics")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes = axes.flatten()
    for i, col in enumerate(play_stats):
        sns.histplot(recent_data[col], bins=15, kde=True, color='salmon', ax=axes[i])
        axes[i].set_title(col)
    plt.tight_layout()
    st.pyplot(fig)

    # Visualize offensive metrics box plots
    st.write("## Offensive Metrics Box Plots")
    fig, axes = plt.subplots(8, 3, figsize=(15, 24))
    axes = axes.flatten()
    for i, col in enumerate(offensive_metrics):
        sns.boxplot(x=recent_data[col], color='skyblue', ax=axes[i])
        axes[i].set_title(col)
    plt.tight_layout()
    st.pyplot(fig)

    # Visualize defensive metrics box plots
    st.write("## Defensive Metrics Box Plots")
    fig, axes = plt.subplots(6, 3, figsize=(15, 18))
    axes = axes.flatten()
    for i, col in enumerate(defensive_metrics):
        sns.boxplot(x=recent_data[col], color='lightgreen', ax=axes[i])
        axes[i].set_title(col)
    plt.tight_layout()
    st.pyplot(fig)

    # Visualize additional play statistics box plots
    st.write("## Play Statistics Box Plots")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes = axes.flatten()
    for i, col in enumerate(play_stats):
        sns.boxplot(x=recent_data[col], color='salmon', ax=axes[i])
        axes[i].set_title(col)
    plt.tight_layout()
    st.pyplot(fig)

    







