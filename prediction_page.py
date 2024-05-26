import streamlit as st
import numpy as np
import pickle
import pandas as pd

def load_model():
    with open('saved_model_v2.pkl', 'rb') as file:
        data_loaded = pickle.load(file)

        if 'missing_go_to_left' in data_loaded['model'].tree.getstate()['nodes'].dtype.names:
            # Transform the model node array to the expected dtype
            nodes = data_loaded['model'].tree.getstate()['nodes']
            expected_dtype = [
                ('left_child', '<i8'), ('right_child', '<i8'), ('feature', '<i8'),
                ('threshold', '<f8'), ('impurity', '<f8'), ('n_node_samples', '<i8'),
                ('weighted_n_node_samples', '<f8')
            ]
            nodes = nodes.astype(expected_dtype)
            data_loaded['model'].tree.setstate({'nodes': nodes})

        return data_loaded

def show_predict_page():
    data_loaded = load_model()

    model_loaded = data_loaded["model"]
    imputer_loaded = data_loaded["imputer"]

    st.title("Predicting Los Angeles Lakers Games")
    st.write("### Provide the following information:")

    features = [
        'pts_home', 'pts_away', 'fgm_home', 'fgm_away', 'fga_home', 'fga_away',
        'fg_pct_home', 'fg_pct_away', 'fg3m_home', 'fg3m_away', 'ftm_home', 'ftm_away',
        'fta_home', 'fta_away', 'ft_pct_home', 'ft_pct_away', 'ast_home', 'ast_away',
        'oreb_home', 'oreb_away', 'reb_home', 'reb_away', 'tov_home', 'tov_away',
        'pf_home', 'pf_away', 'blk_home', 'blk_away', 'stl_home', 'stl_away',
        'dreb_home', 'dreb_away', 'wl_home', 'min', 'fg3a_home', 'fg3_pct_home',
        'plus_minus_home', 'fg3a_away', 'fg3_pct_away', 'plus_minus_away',
        'oreb_diff_home', 'oreb_diff_away'
    ]

    legend = {
        'pts': 'Points scored',
        'fgm': 'Field goals made',
        'fga': 'Field goals attempted',
        'fg_pct': 'Field goal percentage',
        'fg3m': 'Three-point field goals made',
        'ftm': 'Free throws made',
        'fta': 'Free throws attempted',
        'ft_pct': 'Free throw percentage',
        'ast': 'Assists',
        'oreb': 'Offensive rebounds',
        'reb': 'Total rebounds',
        'tov': 'Turnovers',
        'pf': 'Personal fouls',
        'blk': 'Blocks',
        'stl': 'Steals',
        'dreb': 'Defensive rebounds',
        'wl': 'Win or loss (1 for win, 0 for loss)',
        'min': 'Minutes played',
        'fg3a': 'Three-point field goals attempted',
        'fg3_pct': 'Three-point field goal percentage',
        'plus_minus': 'Plus/minus statistic',
        'oreb_diff': 'Difference in offensive rebounds'
    }

    # Example input for LAL playing at home
    home_game = {
        'pts_home': 100, 'pts_away': 98, 'fgm_home': 40, 'fgm_away': 38,
        'fga_home': 97, 'fga_away': 83, 'fg_pct_home': 0.47, 'fg_pct_away': 0.46,
        'fg3m_home': 12, 'fg3m_away': 10, 'ftm_home': 8, 'ftm_away': 12,
        'fta_home': 10, 'fta_away': 15, 'ft_pct_home': 0.8, 'ft_pct_away': 0.8,
        'ast_home': 25, 'ast_away': 22, 'oreb_home': 10, 'oreb_away': 9,
        'reb_home': 50, 'reb_away': 48, 'tov_home': 15, 'tov_away': 14,
        'pf_home': 20, 'pf_away': 22, 'blk_home': 5, 'blk_away': 4,
        'stl_home': 7, 'stl_away': 6, 'dreb_home': 40, 'dreb_away': 39
    }

    # Example input for LAL playing away
    away_game = {
        'pts_home': 98, 'pts_away': 100, 'fgm_home': 38, 'fgm_away': 40,
        'fga_home': 83, 'fga_away': 85, 'fg_pct_home': 0.46, 'fg_pct_away': 0.47,
        'fg3m_home': 10, 'fg3m_away': 12, 'ftm_home': 12, 'ftm_away': 8,
        'fta_home': 15, 'fta_away': 10, 'ft_pct_home': 0.8, 'ft_pct_away': 0.8,
        'ast_home': 22, 'ast_away': 25, 'oreb_home': 9, 'oreb_away': 10,
        'reb_home': 48, 'reb_away': 50, 'tov_home': 14, 'tov_away': 15,
        'pf_home': 22, 'pf_away': 20, 'blk_home': 4, 'blk_away': 5,
        'stl_home': 6, 'stl_away': 7, 'dreb_home': 39, 'dreb_away': 40,
        'fg3a_home': 0, 'fg3a_away': 0, 'fg3_pct_home': 0, 'fg3_pct_away': 0,
        'min': 0, 'plus_minus_home': 0, 'plus_minus_away': 0
    }

    Sample_Game = ['Home', 'Away']
    selected_game = st.selectbox("Select a game to predict:", Sample_Game)

    # Initialize session state for showing stats
    if 'show_stats' not in st.session_state:
        st.session_state.show_stats = False

    # Button to show stats
    if st.button("Show Stats"):
        st.session_state.show_stats = True

    # Show legend
    st.write("### Legend:")
    col1, col2 = st.columns(2)
    legend_items = list(legend.items())
    half = len(legend_items) // 2
    with col1:
        for feature, description in legend_items[:half]:
            st.write(f"**{feature}**: {description}")
    with col2:
        for feature, description in legend_items[half:]:
            st.write(f"**{feature}**: {description}")

    # Show stats if button was clicked
    if st.session_state.show_stats:
        if selected_game == 'Home':
            game_data = home_game
        if selected_game == 'Away':
            game_data = away_game

        st.write(f"### Modify the stats for {selected_game} game:")

        with st.form(key="input_form"):
            inputs = {}
            columns = st.columns(3)
            for i, feature in enumerate(game_data):
                if "pct" in feature:
                    with columns[i % 3]:
                        inputs[feature] = st.number_input(f"{feature}", value=game_data[feature])
                else:
                    with columns[i % 3]:
                        inputs[feature] = st.number_input(f"{feature}", value=game_data[feature])

            submit_button = st.form_submit_button(label="Predict Outcome")

            if submit_button:
                data = pd.DataFrame([inputs])

                def predict_outcome(team_data):
                    model_loaded = data_loaded["model"]
                    imputer_loaded = data_loaded["imputer"]
                    # Ensure the input data has all necessary features
                    for feature in features:
                        if feature not in team_data.columns:
                            team_data[feature] = 0
                    team_data = team_data[features]

                    # Impute missing values in the input data
                    team_data_imputed = imputer_loaded.transform(team_data)

                    # Make prediction
                    prediction = model_loaded.predict(team_data_imputed)
                    return 'Win' if prediction[0] == 1 else 'Lose'

                outcome_home = predict_outcome(data)

                if outcome_home == 'Win':
                    st.markdown(
                        f'<div class="stAlert" style="background-color: #f7e463; color: black;">The predicted outcome for LAL at home is: {outcome_home}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="stAlert" style="background-color: #7d3f98; color: white;">The predicted outcome for LAL at home is: {outcome_home}</div>',
                        unsafe_allow_html=True
                    )


