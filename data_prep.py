import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import datetime
import math


class BusDataPrep:
    """
    Data preparation class for the Bus Arrival Time Prediction project.
    Implements feature engineering and preprocessing steps outlined in the model implementation plan.
    """
    
    def __init__(self, gps_data_path="bus_gps_tracking_data.csv", 
                 stop_data_path="bus_stop_level_data.csv"):
        """
        Initialize with paths to GPS and stop-level data.
        
        Parameters:
        -----------
        gps_data_path : str
            Path to the GPS tracking data CSV file
        stop_data_path : str
            Path to the stop-level data CSV file
        """
        self.gps_data_path = gps_data_path
        self.stop_data_path = stop_data_path
        
        # For storing processed data
        self.gps_df = None
        self.stop_df = None
        self.merged_df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # For preprocessing transformers
        self.numeric_scaler = None
        self.weather_encoder = None
        self.traffic_encoder = None
        
    def load_data(self):
        """Load GPS and stop data, parse timestamps"""
        print("Loading GPS data...")
        self.gps_df = pd.read_csv(self.gps_data_path)
        self.gps_df['timestamp'] = pd.to_datetime(self.gps_df['timestamp'])
        
        print("Loading stop data...")
        self.stop_df = pd.read_csv(self.stop_data_path)
        self.stop_df['timestamp'] = pd.to_datetime(self.stop_df['timestamp'])
        self.stop_df['arrival_time'] = pd.to_datetime(self.stop_df['arrival_time'])
        self.stop_df['departure_time'] = pd.to_datetime(self.stop_df['departure_time'])
        self.stop_df['scheduled_arrival_time'] = pd.to_datetime(self.stop_df['scheduled_arrival_time'])
        
        print(f"Loaded {len(self.gps_df)} GPS records and {len(self.stop_df)} stop records")
        return self
        
    def engineer_temporal_features(self):
        """
        Extract time-based features from timestamp column.
        """
        print("Engineering temporal features...")
        
        # Extract time components
        self.gps_df['hour'] = self.gps_df['timestamp'].dt.hour
        self.gps_df['day_of_week'] = self.gps_df['timestamp'].dt.dayofweek
        self.gps_df['month'] = self.gps_df['timestamp'].dt.month
        
        # Cyclical encoding of time (hour of day)
        self.gps_df['hour_sin'] = np.sin(2 * np.pi * self.gps_df['hour'] / 24)
        self.gps_df['hour_cos'] = np.cos(2 * np.pi * self.gps_df['hour'] / 24)
        
        # Cyclical encoding of day of week
        self.gps_df['day_sin'] = np.sin(2 * np.pi * self.gps_df['day_of_week'] / 7)
        self.gps_df['day_cos'] = np.cos(2 * np.pi * self.gps_df['day_of_week'] / 7)
        
        # Rush hour flag (7-9am and 4-7pm)
        self.gps_df['is_morning_rush'] = ((self.gps_df['hour'] >= 7) & 
                                          (self.gps_df['hour'] < 9)).astype(int)
        self.gps_df['is_evening_rush'] = ((self.gps_df['hour'] >= 16) & 
                                          (self.gps_df['hour'] < 19)).astype(int)
        self.gps_df['is_rush_hour'] = ((self.gps_df['is_morning_rush'] | 
                                       self.gps_df['is_evening_rush'])).astype(int)
        
        # Weekend flag
        self.gps_df['is_weekend'] = (self.gps_df['day_of_week'] >= 5).astype(int)
        
        return self
    
    def engineer_distance_features(self):
        """
        Process distance-related features.
        """
        print("Engineering distance features...")
        
        # Normalize distance to next stop (0-1 scale)
        max_distance = self.gps_df['distance_to_next_stop_km'].max()
        self.gps_df['normalized_distance'] = self.gps_df['distance_to_next_stop_km'] / max_distance
        
        # We'll derive distance from route and stop information
        # Group by bus_id and sort by timestamp to get sequential points
        self.gps_df = self.gps_df.sort_values(['bus_id', 'timestamp'])
        
        # Calculate cumulative distance in trip
        self.gps_df['prev_lat'] = self.gps_df.groupby('bus_id')['latitude'].shift(1)
        self.gps_df['prev_lon'] = self.gps_df.groupby('bus_id')['longitude'].shift(1)
        
        # Calculate distance between consecutive points (when in same trip)
        self.gps_df['point_distance_km'] = self.gps_df.apply(
            lambda row: self._calculate_distance(
                (row['prev_lat'], row['prev_lon']), 
                (row['latitude'], row['longitude'])
            ) if not pd.isna(row['prev_lat']) else 0, 
            axis=1
        )
        
        # Cumulative distance in trip
        self.gps_df['cumulative_distance_km'] = self.gps_df.groupby('bus_id')['point_distance_km'].cumsum()
        
        return self
    
    def _calculate_distance(self, point1, point2):
        """Helper function to calculate distance between two lat/long points in kilometers"""
        # Skip calculation if any point is None/NaN
        if any(pd.isna(p) for p in point1 + point2):
            return 0
            
        # Approximate conversion of lat/long differences to kilometers
        lat_diff = abs(point1[0] - point2[0]) * 111  # 1 degree latitude is approx 111 km
        lon_diff = abs(point1[1] - point2[1]) * 111 * math.cos(math.radians((point1[0] + point2[0]) / 2))
        return math.sqrt(lat_diff**2 + lon_diff**2)
    
    def engineer_weather_traffic_features(self):
        """
        Process weather and traffic related features.
        """
        print("Engineering weather and traffic features...")
        
        # One-hot encode weather conditions
        weather_categories = self.gps_df['weather'].unique()
        self.weather_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        weather_encoded = self.weather_encoder.fit_transform(self.gps_df[['weather']])
        weather_encoded_df = pd.DataFrame(
            weather_encoded, 
            columns=[f'weather_{cat}' for cat in self.weather_encoder.categories_[0]]
        )
        self.gps_df = pd.concat([self.gps_df.reset_index(drop=True), weather_encoded_df], axis=1)
        
        # One-hot encode traffic conditions
        traffic_categories = self.gps_df['traffic'].unique()
        self.traffic_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        traffic_encoded = self.traffic_encoder.fit_transform(self.gps_df[['traffic']])
        traffic_encoded_df = pd.DataFrame(
            traffic_encoded, 
            columns=[f'traffic_{cat}' for cat in self.traffic_encoder.categories_[0]]
        )
        self.gps_df = pd.concat([self.gps_df.reset_index(drop=True), traffic_encoded_df], axis=1)
        
        # Create combined weather-traffic features for severe conditions
        self.gps_df['severe_conditions'] = ((self.gps_df['weather'].isin(['Snow', 'Heavy Rain'])) & 
                                           (self.gps_df['traffic'].isin(['Heavy', 'Gridlock']))).astype(int)
        
        return self
    
    def engineer_historical_features(self):
        """
        Create features based on historical patterns and lag features.
        """
        print("Engineering historical features...")
        
        # Sort data for proper lag creation
        self.gps_df = self.gps_df.sort_values(['bus_id', 'route_id', 'timestamp'])
        
        # Create lag features for delays at previous stops
        self.gps_df['prev_delay'] = self.gps_df.groupby(['bus_id', 'route_id'])['delay_minutes'].shift(1)
        # Fill NA values without using inplace
        self.gps_df['prev_delay'] = self.gps_df['prev_delay'].fillna(0)
        
        # Create rolling average delays for each route
        self.gps_df['route_avg_delay'] = self.gps_df.groupby('route_id')['delay_minutes'].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )
        
        # Cumulative delay in current trip
        self.gps_df['cumulative_delay'] = self.gps_df.groupby(['bus_id', 'route_id'])['delay_minutes'].cumsum()
        
        return self
    
    def handle_missing_values(self):
        """
        Identify and handle missing values in the dataset.
        """
        print("Handling missing values...")
        
        # Check for missing values
        missing_values = self.gps_df.isnull().sum()
        print(f"Missing values before imputation:\n{missing_values[missing_values > 0]}")
        
        # Fill missing values for numeric columns with appropriate defaults
        numeric_cols = self.gps_df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if self.gps_df[col].isnull().sum() > 0:
                # Fix: Use the recommended approach instead of inplace=True
                self.gps_df[col] = self.gps_df[col].fillna(self.gps_df[col].median())
        
        # actual_arrival_time is only filled for arrival events, fill with estimated_arrival_time where null
        if 'actual_arrival_time' in self.gps_df.columns and self.gps_df['actual_arrival_time'].isnull().sum() > 0:
            mask = self.gps_df['actual_arrival_time'].isnull()
            self.gps_df.loc[mask, 'actual_arrival_time'] = self.gps_df.loc[mask, 'estimated_arrival_time']
        
        # Check if we've handled all missing values
        missing_values_after = self.gps_df.isnull().sum()
        print(f"Missing values after imputation:\n{missing_values_after[missing_values_after > 0]}")
        
        return self
    
    def scale_features(self):
        """
        Scale numerical features for model training.
        """
        print("Scaling features...")
        
        # Identify columns to be scaled with StandardScaler
        numeric_cols = [
            'speed_kmh', 'distance_to_next_stop_km', 'passengers_on_board', 
            'fuel_level', 'temperature_c', 'delay_minutes', 'normalized_distance',
            'cumulative_distance_km', 'prev_delay', 'route_avg_delay', 'cumulative_delay'
        ]
        
        # Filter to only include columns that exist in the dataframe
        numeric_cols = [col for col in numeric_cols if col in self.gps_df.columns]
        
        # Fit scaler on numeric columns
        self.numeric_scaler = StandardScaler()
        scaled_features = self.numeric_scaler.fit_transform(self.gps_df[numeric_cols])
        
        # Create a new DataFrame with scaled values
        scaled_df = pd.DataFrame(
            scaled_features, 
            columns=[f'{col}_scaled' for col in numeric_cols],
            index=self.gps_df.index
        )
        
        # Add scaled features to original dataframe
        self.gps_df = pd.concat([self.gps_df, scaled_df], axis=1)
        
        return self
    
    def prepare_target_variable(self):
        """
        Prepare the target variable for prediction.
        """
        print("Preparing target variable...")
        
        # Define target (we'll predict delay_minutes as our main target)
        # For records with actual_arrival_time, we'll calculate the actual delay
        mask = ~pd.isna(self.gps_df['actual_arrival_time'])
        if mask.sum() > 0:
            self.gps_df.loc[mask, 'actual_delay'] = self.gps_df.loc[mask, 'delay_minutes']
        else:
            # If no actual arrival times, use the existing delay_minutes
            self.gps_df['actual_delay'] = self.gps_df['delay_minutes']
        
        # Add target for arrival time prediction as well
        self.gps_df['arrival_time_target'] = (
            pd.to_datetime(self.gps_df['estimated_arrival_time']) - 
            pd.to_datetime(self.gps_df['timestamp'])
        ).dt.total_seconds() / 60  # Convert to minutes
        
        return self
    
    def split_data(self, test_size=0.15, val_size=0.15, random_state=42):
        """
        Split data into training, validation, and test sets.
        
        Parameters:
        -----------
        test_size : float
            Proportion of data to use for test set
        val_size : float
            Proportion of data to use for validation set
        random_state : int
            Random seed for reproducibility
        """
        print("Splitting data into train/val/test sets...")
        
        # Define features and target
        features = [col for col in self.gps_df.columns if col not in [
            'record_id', 'timestamp', 'estimated_arrival_time', 'actual_arrival_time',
            'weather', 'traffic', 'actual_delay', 'arrival_time_target'
        ]]
        
        # We'll predict both delay and arrival time
        X = self.gps_df[features]
        y_delay = self.gps_df['actual_delay']
        y_arrival = self.gps_df['arrival_time_target']
        
        # First split: separate out test set
        X_temp, self.X_test, y_delay_temp, self.y_delay_test, y_arrival_temp, self.y_arrival_test = train_test_split(
            X, y_delay, y_arrival, test_size=test_size, random_state=random_state
        )
        
        # Second split: create train and validation sets
        val_ratio = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_delay_train, self.y_delay_val, self.y_arrival_train, self.y_arrival_val = train_test_split(
            X_temp, y_delay_temp, y_arrival_temp, test_size=val_ratio, random_state=random_state
        )
        
        print(f"Train set: {len(self.X_train)} samples")
        print(f"Validation set: {len(self.X_val)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        
        return self
    
    def prepare_sequence_data(self, seq_length=10):
        """
        Prepare sequential data for LSTM/GRU models.
        
        Parameters:
        -----------
        seq_length : int
            Number of time steps in each sequence
        """
        print(f"Preparing sequence data with length {seq_length}...")
        
        # Sort data by bus_id and timestamp to ensure proper sequence
        sorted_df = self.gps_df.sort_values(['bus_id', 'timestamp'])
        
        # Identify features for sequences
        sequence_features = [col for col in sorted_df.columns if col.endswith('_scaled')]
        sequence_features += [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
            'is_rush_hour', 'is_weekend'
        ]
        sequence_features += [col for col in sorted_df.columns if col.startswith('weather_')]
        sequence_features += [col for col in sorted_df.columns if col.startswith('traffic_')]
        
        # Filter to only include columns that exist
        sequence_features = [col for col in sequence_features if col in sorted_df.columns]
        
        # Create sequences for each bus
        X_sequences = []
        y_delay_sequences = []
        y_arrival_sequences = []
        
        # Group by bus_id to create sequences
        for bus_id, group in sorted_df.groupby('bus_id'):
            if len(group) < seq_length:
                continue
                
            # Create sequences
            for i in range(len(group) - seq_length):
                X_sequences.append(group[sequence_features].iloc[i:i+seq_length].values)
                y_delay_sequences.append(group['actual_delay'].iloc[i+seq_length])
                y_arrival_sequences.append(group['arrival_time_target'].iloc[i+seq_length])
        
        # Convert to numpy arrays
        self.X_sequences = np.array(X_sequences)
        self.y_delay_sequences = np.array(y_delay_sequences)
        self.y_arrival_sequences = np.array(y_arrival_sequences)
        
        # Split into train/val/test
        total_sequences = len(X_sequences)
        train_idx = int(total_sequences * 0.7)
        val_idx = int(total_sequences * 0.85)
        
        self.X_seq_train = self.X_sequences[:train_idx]
        self.y_delay_seq_train = self.y_delay_sequences[:train_idx]
        self.y_arrival_seq_train = self.y_arrival_sequences[:train_idx]
        
        self.X_seq_val = self.X_sequences[train_idx:val_idx]
        self.y_delay_seq_val = self.y_delay_sequences[train_idx:val_idx]
        self.y_arrival_seq_val = self.y_arrival_sequences[train_idx:val_idx]
        
        self.X_seq_test = self.X_sequences[val_idx:]
        self.y_delay_seq_test = self.y_delay_sequences[val_idx:]
        self.y_arrival_seq_test = self.y_arrival_sequences[val_idx:]
        
        print(f"Created {len(X_sequences)} sequences")
        print(f"Sequence train set: {len(self.X_seq_train)} samples")
        print(f"Sequence validation set: {len(self.X_seq_val)} samples")
        print(f"Sequence test set: {len(self.X_seq_test)} samples")
        
        return self
    
    def save_processed_data(self, output_dir="processed_data"):
        """
        Save processed data to files for later use.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save processed data files
        """
        import os
        
        print(f"Saving processed data to {output_dir}...")
        
        # Create directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save processed dataframe
        self.gps_df.to_csv(f"{output_dir}/processed_gps_data.csv", index=False)
        
        # Save train/val/test sets
        self.X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
        self.X_val.to_csv(f"{output_dir}/X_val.csv", index=False)
        self.X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
        
        # Save target variables
        pd.DataFrame({'delay': self.y_delay_train}).to_csv(f"{output_dir}/y_delay_train.csv", index=False)
        pd.DataFrame({'delay': self.y_delay_val}).to_csv(f"{output_dir}/y_delay_val.csv", index=False)
        pd.DataFrame({'delay': self.y_delay_test}).to_csv(f"{output_dir}/y_delay_test.csv", index=False)
        
        pd.DataFrame({'arrival': self.y_arrival_train}).to_csv(f"{output_dir}/y_arrival_train.csv", index=False)
        pd.DataFrame({'arrival': self.y_arrival_val}).to_csv(f"{output_dir}/y_arrival_val.csv", index=False)
        pd.DataFrame({'arrival': self.y_arrival_test}).to_csv(f"{output_dir}/y_arrival_test.csv", index=False)
        
        # Save sequence data
        np.save(f"{output_dir}/X_seq_train.npy", self.X_seq_train)
        np.save(f"{output_dir}/X_seq_val.npy", self.X_seq_val)
        np.save(f"{output_dir}/X_seq_test.npy", self.X_seq_test)
        
        np.save(f"{output_dir}/y_delay_seq_train.npy", self.y_delay_seq_train)
        np.save(f"{output_dir}/y_delay_seq_val.npy", self.y_delay_seq_val)
        np.save(f"{output_dir}/y_delay_seq_test.npy", self.y_delay_seq_test)
        
        np.save(f"{output_dir}/y_arrival_seq_train.npy", self.y_arrival_seq_train)
        np.save(f"{output_dir}/y_arrival_seq_val.npy", self.y_arrival_seq_val)
        np.save(f"{output_dir}/y_arrival_seq_test.npy", self.y_arrival_seq_test)
        
        print("Data saved successfully!")
        return self
    
    def run_full_pipeline(self):
        """
        Run the full data preparation pipeline.
        """
        (self.load_data()
             .engineer_temporal_features()
             .engineer_distance_features()
             .engineer_weather_traffic_features()
             .engineer_historical_features()
             .handle_missing_values()
             .scale_features()
             .prepare_target_variable()
             .split_data()
             .prepare_sequence_data()
             .save_processed_data())
        
        print("Data preparation pipeline completed successfully!")
        return self


if __name__ == "__main__":
    print("Starting Bus Arrival Time Prediction data preparation...")
    data_prep = BusDataPrep()
    data_prep.run_full_pipeline()
    print("Data preparation complete!") 