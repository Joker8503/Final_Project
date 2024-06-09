# NBA Game Outcome Prediction for Los Angeles Lakers (LAL)

## Introduction
In this project, I developed a predictive model to forecast the outcomes of NBA games for the Los Angeles Lakers (LAL) using an ExtraTreesClassifier. Key performance metrics and features such as points, shooting percentages, and assists were analyzed, and the model was tuned using GridSearchCV. The final model and imputer were saved for future predictions, ensuring robust performance through cross-validation and handling imbalanced data with SMOTE.

[Click here to view the deployed model](https://lalgameprediction.streamlit.app/)

## Data Source
The data used for this project was sourced from [Kaggle](https://www.kaggle.com/datasets/wyattowalsh/basketball).

### Dataset License
The dataset is provided under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license. You can view the full license [here](https://creativecommons.org/licenses/by-sa/4.0/).

## Technologies
- Python
- Scikit-learn
- Pandas
- Streamlit
- SMOTE (Synthetic Minority Over-sampling Technique)

## Installation and Setup
To run this project locally, follow these steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/Joker8503/NBA-LAL-Predictor.git
    cd NBA-LAL-Predictor
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Project Structure
- `data/`: Contains the dataset used for training and testing.
- `notebooks/`: Jupyter notebooks with data analysis, model training, and evaluation.
- `app.py`: Streamlit app for interactive model prediction.
- `requirements.txt`: List of required Python packages.

## Key Insights
- **Performance Metrics:** Points, shooting percentages, and assists were critical for predicting game outcomes.
- **Home vs. Away Performance:** Home-court advantage significantly impacted team performance.
- **Feature Engineering:** Added `is_home` feature to capture home-court advantage.

## Evaluation Metrics
- **Accuracy:** Overall correctness of the model.
- **Precision and Recall:** Balanced the trade-off between identifying relevant instances and minimizing false positives.
- **F1-Score:** Harmonic mean of precision and recall, providing a balanced measure.
- **Confusion Matrix:** Visualized true positives, false positives, true negatives, and false negatives.

## Future Work
- Incorporate additional features like player statistics.
- Explore other machine learning algorithms to improve prediction accuracy.
- Implement real-time data updates for predictions.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
