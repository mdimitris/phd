import pandas as pd
import dask.dataframe as dd
import xgboost as xgb
import os
import numpy as np

class xgBoostFill:
    """
    A class to perform missing data imputation using XGBoost.
    It trains a separate XGBoost model for each target column with missing values,
    using other specified features to predict the missing data.
    """
    def __init__(self, target_columns, features, random_state=42):
        self.target_columns = target_columns
        self.features = features
        self.random_state = random_state
        self.models = {}

    def fit(self, data):
        """
        Trains an XGBoost regressor for each target column with missing values.
        
        This method is designed to be called on a pandas DataFrame (a sample of your
        full Dask DataFrame).

        Args:
            data (pd.DataFrame): The DataFrame to train on. This should be a representative
                                 sample of your full dataset.
        """
        print("Training XGBoost models for imputation...")
        for col in self.target_columns:
            # Drop the current target column from the feature list for this specific model
            current_features = [f for f in self.features if f != col]
            
            # Filter out rows with missing values for the current target column
            train_data = data.dropna(subset=[col])
            
            if train_data.empty:
                print(f"Skipping training for '{col}' due to no complete cases.")
                continue

            # Prepare features and target for training
            X_train = train_data[current_features]
            y_train = train_data[col]
            
            # Initialize and train the XGBoost regressor
            model = xgb.XGBRegressor(
                objective='reg:squarederror', 
                n_estimators=500, 
                learning_rate=0.01,
                max_depth=6, 
                random_state=self.random_state,
                n_jobs=-1  # Use all available CPU cores
            )
            model.fit(X_train, y_train)
            self.models[col] = model
        print("Models have been trained for all specified vital signs columns.")
        return self

    def transform(self, data):
        """
        Uses the trained models to fill missing values in the target columns.
        
        This method is designed to be called on a pandas DataFrame partition by Dask.

        Args:
            data (pd.DataFrame): The DataFrame partition to be transformed.
        
        Returns:
            pd.DataFrame: A new DataFrame with the missing values filled.
        """
        filled_data = data.copy()
        
        for col, model in self.models.items():
            # Drop the current target column from the feature list for prediction
            current_features = [f for f in self.features if f != col]
            
            # Identify rows with missing values to be predicted
            missing_indices = filled_data[filled_data[col].isnull()].index
            
            if not missing_indices.empty:
                # Prepare features for prediction
                X_predict = filled_data.loc[missing_indices, current_features]
                
                # Predict the missing values
                predictions = model.predict(X_predict)
                
                # Fill the missing values in the copied DataFrame
                filled_data.loc[missing_indices, col] = predictions
                # print(f"Filled {len(missing_indices)} missing values in column '{col}' for this partition.")
            else:
                # print(f"No missing values to fill in column '{col}' for this partition.")
                pass
                
        return filled_data

    def fit_transform(self, data):
        """
        Fits the models and then transforms the data in one step.
        """
        self.fit(data)
        return self.transform(data)