import dask.dataframe as dd
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from dask.distributed import Client
import os
import warnings


# Suppress Dask performance warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="dask")

class LSTMModel(nn.Module):
    """A simple LSTM model for time-series prediction."""
    def __init__(self, input_size, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class LSTMImputer:
    """
    An imputer that uses a PyTorch LSTM model to fill missing values in a target column
    of a Dask DataFrame based on a set of feature columns.
    """
    def __init__(self, target_col, feature_cols, seq_len=12, epochs=5, batch_size=32, lr=0.001, hidden_size=50):
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.all_cols = feature_cols + target_col
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden_size = hidden_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def _create_sequences(self, data_X, data_y):
        """Creates sequences for LSTM training."""
        X, y = [], []
        for i in range(len(data_X) - self.seq_len):
            X.append(data_X[i:(i + self.seq_len)])
            y.append(data_y[i + self.seq_len])
        return np.array(X), np.array(y)

    def fit(self, ddf):
        """
        Fits the LSTM model on the non-missing data.
        
        Args:
            ddf (dask.DataFrame): The Dask DataFrame to train on.
        """
        print("Starting model training...")
        
        # For large datasets, training on a sample is more practical.
        # This computes a fraction of the data into a pandas DataFrame.
        print("Fetching and preparing training data sample...")
        df_train_sample = ddf[self.all_cols].dropna().sample(frac=0.1, random_state=42).compute()
        
        if len(df_train_sample) < self.seq_len * 2:
            raise ValueError("Training sample is too small. Increase sample fraction or check data for NaNs.")

        # Scale features and target
        X_scaled = self.scaler_X.fit_transform(df_train_sample[self.feature_cols])
        y_scaled = self.scaler_y.fit_transform(df_train_sample[self.target_col])

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.from_numpy(X_seq).float().to(self.device)
        y_train_tensor = torch.from_numpy(y_seq).float().to(self.device)

        # Initialize model, loss, and optimizer
        self.model = LSTMModel(input_size=len(self.feature_cols), hidden_size=self.hidden_size).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Training loop
        print(f"Training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            self.model.train()
            for i in range(0, len(X_train_tensor), self.batch_size):
                X_batch = X_train_tensor[i:i+self.batch_size]
                y_batch = y_train_tensor[i:i+self.batch_size]

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            
            print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')
        
        print("Model training complete.")
        return self

    def transform(self, ddf):
        """
        Fills missing values in the target column of the DataFrame.
        This method handles both Dask and pandas DataFrames.

        Args:
            ddf (dask.DataFrame or pandas.DataFrame): The DataFrame to impute.
        
        Returns:
            dask.DataFrame or pandas.DataFrame: DataFrame with the target column imputed, matching the input type.
        """
        if self.model is None:
            raise RuntimeError("Imputer has not been fitted. Please call .fit() first.")
        
        print("Starting imputation transform...")
        self.model.eval()

        def impute_partition(df):
            """
            The function to apply to each overlapping partition or a whole pandas DataFrame.
            This version uses integer positions (iloc) to be robust against duplicate indices.
            """
            df = df.copy()
            target_col_name = self.target_col[0]
            
            # Find integer row positions of NaNs in the target column
            missing_locs = np.where(df[target_col_name].isna())[0]

            for loc in missing_locs:
                # loc is now a guaranteed integer position
                if loc < self.seq_len:
                    continue

                # Get the sequence of features before the missing value using iloc
                feature_sequence_pd = df[self.feature_cols].iloc[loc - self.seq_len : loc]

                # FIX: If the feature window itself contains NaNs, apply a forward fill,
                # then a backfill. This ensures the LSTM model always receives a complete sequence.
                if feature_sequence_pd.isnull().values.any():
                    feature_sequence_pd = feature_sequence_pd.interpolate(method='linear').ffill().bfill()

                # If NaNs still exist (e.g., an entire column in the window was NaN), we must skip.
                if feature_sequence_pd.isnull().values.any():
                    continue
                
                # Scale the features
                feature_sequence_scaled = self.scaler_X.transform(feature_sequence_pd)
                
                # Convert to tensor and add batch dimension
                sequence_tensor = torch.from_numpy(feature_sequence_scaled).float().unsqueeze(0).to(self.device)

                # Predict with the model
                with torch.no_grad():
                    predicted_scaled = self.model(sequence_tensor)
                
                # Inverse transform the prediction to the original scale
                predicted_value = self.scaler_y.inverse_transform(predicted_scaled.cpu().numpy())[0][0]
                
                # Fill the NaN value using the integer position (iloc)
                target_col_loc = df.columns.get_loc(target_col_name)
                df.iloc[loc, target_col_loc] = predicted_value
            
            return df

        # Check if the input is a pandas DataFrame or a Dask DataFrame
        if isinstance(ddf, pd.DataFrame):
            # If it's a pandas DataFrame, apply the imputation function directly
            print("Input is a pandas DataFrame. Applying imputation directly.")
            imputed_result = impute_partition(ddf)
        elif isinstance(ddf, dd.DataFrame):
            # If it's a Dask DataFrame, use map_overlap for parallel processing
            print("Input is a Dask DataFrame. Using map_overlap for parallel imputation.")
            imputed_result = ddf.map_overlap(
                impute_partition,
                before=self.seq_len,
                after=0,
                meta=ddf._meta
            )
        else:
            raise TypeError(f"Input must be a pandas or Dask DataFrame, but got {type(ddf)}")
        
        print("Imputation transform complete.")
        return imputed_result
    
    def save(self, ddf, path):
        """
        Saves the Dask DataFrame to a Parquet file.

        Args:
            ddf (dask.DataFrame): The DataFrame to save.
            path (str): The directory path to save the Parquet file.
        """
        print(f"Saving imputed DataFrame to {path}...")
        if not os.path.exists(path):
            os.makedirs(path)
        ddf.to_parquet(path, write_index=False)
        print("Save complete.")
