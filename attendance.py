import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer

from game import Game
from preprocessing import Preprocessing

# source data csv filepath
DATA_FILENAME = Path(__file__).parent/'data/DukeAttendanceV10.csv'

class Attendance():
    @staticmethod
    def get_conference_opponents():
        return ['Boston College', 'California', 'Clemson', 'Duke', 'Florida St.', 'Georgia Tech', 'Louisville', 'Miami', 'North Carolina', 'North Carolina St.', 'Notre Dame', 'Pittsburgh', 'SMU', 'Stanford', 'Syracuse', 'Virginia', 'Virginia Tech', 'Wake Forest']

    @staticmethod
    def get_2025_home_opponents(include_noncon=True):
        if include_noncon:
            return ['Illinois', 'Georgia Tech', 'North Carolina St.', 'Virginia', 'Wake Forest']
        else:
            return ['Georgia Tech', 'North Carolina St.', 'Virginia', 'Wake Forest']

    @staticmethod
    def prepare_model():
        # 1. Load the data
        df = pd.read_csv(DATA_FILENAME)

        # 2. Filter for home games with actual attendance
        df = df[(df['Site'] == 'Home') & (df['AttNum'].notnull())]
        df = Preprocessing.add_day_of_year(df)

        # Drop future years and certain previous years
        df = df[df['Year'] < 2025]
        df = df[df['COVID_Limit'] == 0]
        # df = df[df['Year'] > 2005]

        # Determine the minimum OppFPI_PrevYear to use for imputation
        fpi_min = df['OppFPI_PrevYear'].min()

        # 3. Select features and target
        features = ['Year', 'OppFPI_PrevYear', 'FPI_Diff_PrevYear', 'DayOfYear', 'Start_Time', 'Temp', 'ESPN_WinPred', 'RegularSeason_PrevYear', 'SRS_PrevYear', 'SOS_PrevYear', 'OppCityDist',
                    'OppName', 'OnSaturday', 'Renovated', 'MaxCapacity', 'First_Game', 'First_Home_Game', 'Elko', 'Cutcliffe', 'Diaz', 'Rain', 'Bowl_PrevYear', 'DukeRankedGametime', 'OppRankedGametime', '1stSeedQB', 'SchoolBreak', 'NatlHoliday', 'ThanksgivingWeekend', 'LaborDayWeekend', 'Undefeated_All', 'NC_Opponent', 'TobaccoRoadGame']
        target = 'AttNum'

        X = df[features]
        y = df[target]

        # Identify categorical vs numeric columns
        numeric_features = ['Year', 'OppFPI_PrevYear', 'FPI_Diff_PrevYear', 'DayOfYear', 'Start_Time', 'Temp', 'ESPN_WinPred', 'RegularSeason_PrevYear', 'SRS_PrevYear', 'SOS_PrevYear', 'Bowl_PrevYear', 'OppCityDist']
        categorical_features = ['OppName', 'OnSaturday', 'Renovated', 'MaxCapacity', 'First_Game', 'First_Home_Game', 'Elko', 'Cutcliffe', 'Diaz', 'Rain', 'Bowl_PrevYear', 'DukeRankedGametime', 'OppRankedGametime', '1stSeedQB', 'SchoolBreak', 'NatlHoliday', 'ThanksgivingWeekend', 'LaborDayWeekend', 'Undefeated_All', 'NC_Opponent', 'TobaccoRoadGame']

        # 4. Preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        # 5. Build a pipeline with a model
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ])

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

        # Optionally run a grid search to optimize hyperparameters
        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [None, 10, 20]
        }

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        print(f"Test MAE: {mae:.2f}")
        print(f"Test RMSE: {rmse:.2f}")

        # 6. Check feature importance (for Random Forest)
        final_model = best_model.named_steps['regressor']
        # To check importance after pipeline: 
        feature_names = (numeric_features + 
                        list(best_model.named_steps['preprocessor'].transformers_[1][1]
                            .named_steps['onehot']
                            .get_feature_names_out(categorical_features)))

        importances = final_model.feature_importances_
        feature_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        print(feature_importance)

        return

    @staticmethod
    def predict_2025():
        return

    @staticmethod
    def test_prediction():
        # Filename of csv containing attendance data
        filename = Path(__file__).parent/'data/DukeAttendanceV10.csv'

        # 1. Load the data
        df = pd.read_csv(filename)

        # 2. Filter for home games with actual attendance
        df = df[(df['Site'] == 'Home') & (df['AttNum'].notnull())]

        # Drop future years and certain previous years
        df = df[df['Year'] < 2025]
        # df = df[df['Year'] > 2005]

        # Determine the minimum OppFPI_PrevYear to use for imputation
        fpi_min = df['OppFPI_PrevYear'].min()

        # 3. Select features and target
        features = ['OppFPI_PrevYear', 'OppCityDist', 'MaxCapacity',  # 'Month', 'Start_Time',
                    'ThanksgivingWeekend', 'LaborDayWeekend', 'UNC_Game', 'OppName']
        target = 'AttNum'

        X = df[features]
        y = df[target]

        # Identify categorical vs numeric columns
        numeric_features = ['OppFPI_PrevYear', 'OppCityDist', 'MaxCapacity']  # ['Month', 'Start_Time']
        categorical_features = ['ThanksgivingWeekend', 'LaborDayWeekend', 'UNC_Game', 'OppName']

        # 4. Preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        # 5. Build a pipeline with a model
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ])

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

        # Optionally run a grid search to optimize hyperparameters
        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [None, 10, 20]
        }

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        print(f"Test MAE: {mae:.2f}")
        print(f"Test RMSE: {rmse:.2f}")

        # 6. Check feature importance (for Random Forest)
        final_model = best_model.named_steps['regressor']
        # To check importance after pipeline: 
        feature_names = (numeric_features + 
                        list(best_model.named_steps['preprocessor'].transformers_[1][1]
                            .named_steps['onehot']
                            .get_feature_names_out(categorical_features)))

        importances = final_model.feature_importances_
        feature_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        print(feature_importance)

        # 7. Using the final model for 2025 predictions
        # Filter the original df or another df with 2025 future games (no AttNum)
        df_2025 = pd.read_csv(filename)
        df_2025 = df_2025[(df_2025['Site'] == 'Home') & (df_2025['Year'] == 2025)]

        X_2025 = df_2025[features]
        predicted_att = best_model.predict(X_2025)

        df_2025['PredictedAttendance'] = predicted_att
        print(df_2025[['Year', 'OppName', 'PredictedAttendance']])

        return(df_2025[['Year', 'OppName', 'PredictedAttendance']])
