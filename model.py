# Importing essential libraries
import pandas as pd
import numpy as np
from datetime import datetime
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse

# Loading the dataset
df = pd.read_csv('data/data.csv')
columns_to_remove = [
    'mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker'
]
df.drop(labels=columns_to_remove, axis=1, inplace=True)

consistent_teams = [
    'Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
    'Delhi Daredevils', 'Sunrisers Hyderabad'
]

# Keeping only consistent teams
df = df[(df['bat_team'].isin(consistent_teams))
        & (df['bowl_team'].isin(consistent_teams))]

df['bat_team'].unique()
df = df[df['overs'] >= 5.0]
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

# Get correlation of all the features of the dataset
corr_matrix = df.corr()
top_corr_features = corr_matrix.index

encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])
encoded_df.columns

encoded_df.head()

encoded_df = encoded_df[[
    'date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils',
    'bat_team_Kings XI Punjab', 'bat_team_Kolkata Knight Riders',
    'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
    'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
    'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils',
    'bowl_team_Kings XI Punjab', 'bowl_team_Kolkata Knight Riders',
    'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
    'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
    'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total'
]]

# Splitting the data into train and test set
X_train = encoded_df.drop(labels='total',
                          axis=1)[encoded_df['date'].dt.year <= 2016]
X_test = encoded_df.drop(labels='total',
                         axis=1)[encoded_df['date'].dt.year >= 2017]

y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values

# Removing the 'date' column
X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date', axis=True, inplace=True)

print("Training set: {} and Test set: {}".format(X_train.shape, X_test.shape))


def getUniqueValues():
    return {
        "batting_team": [
            "Chennai Super Kings", "Delhi Daredevils", "Kings XI Punjab",
            "Kolkata Knight Riders", "Mumbai Indians", "Rajasthan Royals",
            "Royal Challengers Bangalore", "Sunrisers Hyderabad"
        ],
        "bowling_team": [
            "Chennai Super Kings", "Delhi Daredevils", "Kings XI Punjab",
            "Kolkata Knight Riders", "Mumbai Indians", "Rajasthan Royals",
            "Royal Challengers Bangalore", "Sunrisers Hyderabad"
        ]
    }


# Predictions
# • Model trained on the data from IPL Seasons 1 to 9 ie: (2008 to 2016)
# • Model tested on data from IPL Season 10 ie: (2017)
# • Model predicts on data from IPL Seasons 11 to 12 ie: (2018 to 2019)
def predict(model, batting_team, bowling_team, overs, runs, wickets,
            runs_in_prev_5, wickets_in_prev_5):
    temp_array = list()

    # Batting Team
    if batting_team == 'Chennai Super Kings':
        temp_array = temp_array + [1, 0, 0, 0, 0, 0, 0, 0]
    elif batting_team == 'Delhi Daredevils':
        temp_array = temp_array + [0, 1, 0, 0, 0, 0, 0, 0]
    elif batting_team == 'Kings XI Punjab':
        temp_array = temp_array + [0, 0, 1, 0, 0, 0, 0, 0]
    elif batting_team == 'Kolkata Knight Riders':
        temp_array = temp_array + [0, 0, 0, 1, 0, 0, 0, 0]
    elif batting_team == 'Mumbai Indians':
        temp_array = temp_array + [0, 0, 0, 0, 1, 0, 0, 0]
    elif batting_team == 'Rajasthan Royals':
        temp_array = temp_array + [0, 0, 0, 0, 0, 1, 0, 0]
    elif batting_team == 'Royal Challengers Bangalore':
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 1, 0]
    elif batting_team == 'Sunrisers Hyderabad':
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 1]

    # Bowling Team
    if bowling_team == 'Chennai Super Kings':
        temp_array = temp_array + [1, 0, 0, 0, 0, 0, 0, 0]
    elif bowling_team == 'Delhi Daredevils':
        temp_array = temp_array + [0, 1, 0, 0, 0, 0, 0, 0]
    elif bowling_team == 'Kings XI Punjab':
        temp_array = temp_array + [0, 0, 1, 0, 0, 0, 0, 0]
    elif bowling_team == 'Kolkata Knight Riders':
        temp_array = temp_array + [0, 0, 0, 1, 0, 0, 0, 0]
    elif bowling_team == 'Mumbai Indians':
        temp_array = temp_array + [0, 0, 0, 0, 1, 0, 0, 0]
    elif bowling_team == 'Rajasthan Royals':
        temp_array = temp_array + [0, 0, 0, 0, 0, 1, 0, 0]
    elif bowling_team == 'Royal Challengers Bangalore':
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 1, 0]
    elif bowling_team == 'Sunrisers Hyderabad':
        temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 1]

    # Overs, Runs, Wickets, Runs_in_prev_5, Wickets_in_prev_5
    temp_array = temp_array + [
        overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5
    ]
    temp_array = np.array([temp_array])
    score = int(model.predict(temp_array)[0])

    return [score - 10, score + 5]


if __name__ == "__main__":
    linear_regressor = LinearRegression()
    linear_regressor.fit(X_train, y_train)

    dump(linear_regressor, "model.joblib")

    y_pred_lr = linear_regressor.predict(X_test)

    print("---- Linear Regression - Model Evaluation ----")
    print("Mean Absolute Error (MAE): {}".format(mae(y_test, y_pred_lr)))
    print("Mean Squared Error (MSE): {}".format(mse(y_test, y_pred_lr)))
    print("Root Mean Squared Error (RMSE): {}".format(
        np.sqrt(mse(y_test, y_pred_lr))))
