import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import uuid

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_bus_gps_data(num_records=500000, num_buses=100, num_routes=15, 
                          city_center=(40.7128, -74.0060)):  # NYC coordinates as default
    """
    Generate synthetic bus GPS tracking data
    
    Parameters:
    -----------
    num_records : int
        Number of records to generate
    num_buses : int
        Number of unique buses in the system
    num_routes : int
        Number of unique routes
    city_center : tuple
        (latitude, longitude) of city center
    
    Returns:
    --------
    DataFrame with synthetic bus GPS tracking data
    """
    
    # Create bus IDs
    bus_ids = [f"BUS_{str(i).zfill(4)}" for i in range(1, num_buses + 1)]
    
    # Create route IDs and details
    route_ids = [f"R{str(i).zfill(2)}" for i in range(1, num_routes + 1)]
    
    # Create route details (start and end points)
    route_details = {}
    for route_id in route_ids:
        # Generate random start and end points within reasonable distance of city center
        start_lat = city_center[0] + np.random.uniform(-0.1, 0.1)
        start_lon = city_center[1] + np.random.uniform(-0.1, 0.1)
        end_lat = city_center[0] + np.random.uniform(-0.1, 0.1)
        end_lon = city_center[1] + np.random.uniform(-0.1, 0.1)
        
        # Ensure start and end points are different
        while abs(start_lat - end_lat) < 0.02 and abs(start_lon - end_lon) < 0.02:
            end_lat = city_center[0] + np.random.uniform(-0.1, 0.1)
            end_lon = city_center[1] + np.random.uniform(-0.1, 0.1)
        
        route_details[route_id] = {
            "start": (start_lat, start_lon),
            "end": (end_lat, end_lon),
            "avg_duration_minutes": np.random.randint(20, 90),
            "stops": np.random.randint(5, 20)
        }
    
    # Weather conditions
    weather_conditions = ["Clear", "Cloudy", "Light Rain", "Heavy Rain", "Snow", "Fog"]
    weather_probabilities = [0.50, 0.25, 0.10, 0.05, 0.05, 0.05]

    # Traffic conditions
    traffic_conditions = ["Light", "Moderate", "Heavy", "Gridlock"]
    traffic_probabilities = [0.4, 0.3, 0.2, 0.1]
    
    # Day types (weekday, weekend)
    day_types = ["Weekday", "Weekend"]
    
    # Initialize empty lists for each column
    data = {
        "record_id": [],
        "timestamp": [],
        "bus_id": [],
        "route_id": [],
        "latitude": [],
        "longitude": [],
        "speed_kmh": [],
        "heading": [],
        "passengers": [],
        "fuel_level": [],
        "temperature_c": [],
        "weather": [],
        "traffic": [],
        "day_type": [],
        "is_delayed": [],
        "delay_minutes": [],
        "is_breakdown": [],
        "driver_id": []
    }
    
    # Assign drivers to buses
    driver_assignments = {}
    for bus_id in bus_ids:
        driver_assignments[bus_id] = f"D{str(random.randint(1000, 9999))}"
    
    # Bus assignments to routes
    bus_to_route = {}
    for bus_id in bus_ids:
        bus_to_route[bus_id] = random.choice(route_ids)
    
    # Generate start timestamps (covering a 30-day period)
    start_date = datetime(2023, 1, 1)
    
    # Generate the data
    for i in range(num_records):
        if i % 50000 == 0:
            print(f"Generating record {i}/{num_records}")
        
        # Randomly pick a bus
        bus_id = random.choice(bus_ids)
        
        # Get the route for this bus
        route_id = bus_to_route[bus_id]
        route = route_details[route_id]
        
        # Generate a random timestamp within 30 days
        days_offset = np.random.randint(0, 30)
        hours_offset = np.random.randint(0, 24)
        minutes_offset = np.random.randint(0, 60)
        seconds_offset = np.random.randint(0, 60)
        
        timestamp = start_date + timedelta(days=days_offset, hours=hours_offset, 
                                          minutes=minutes_offset, seconds=seconds_offset)
        
        # Determine day type
        if timestamp.weekday() < 5:  # Monday=0, Sunday=6
            day_type = "Weekday"
        else:
            day_type = "Weekend"
        
        # Generate random position between start and end
        progress = np.random.random()  # 0 to 1, how far along the route
        
        latitude = route["start"][0] + progress * (route["end"][0] - route["start"][0])
        longitude = route["start"][1] + progress * (route["end"][1] - route["start"][1])
        
        # Add some noise to the position
        latitude += np.random.normal(0, 0.001)
        longitude += np.random.normal(0, 0.001)
        
        # Generate random speed (0 when stopped, otherwise 10-70 km/h)
        if np.random.random() < 0.05:  # 5% chance the bus is stopped
            speed = 0
        else:
            # Speed varies by traffic and weather
            base_speed = np.random.uniform(20, 60)
            
            # Traffic affects speed
            traffic = np.random.choice(traffic_conditions, p=traffic_probabilities)
            if traffic == "Light":
                speed_multiplier = np.random.uniform(0.9, 1.1)
            elif traffic == "Moderate":
                speed_multiplier = np.random.uniform(0.7, 0.9)
            elif traffic == "Heavy":
                speed_multiplier = np.random.uniform(0.5, 0.7)
            else:  # Gridlock
                speed_multiplier = np.random.uniform(0.1, 0.3)
            
            speed = base_speed * speed_multiplier
        
        # Calculate heading (0-359 degrees)
        if route["end"][0] > route["start"][0]:
            base_heading = 45 if route["end"][1] > route["start"][1] else 135
        else:
            base_heading = 315 if route["end"][1] > route["start"][1] else 225
        
        heading = (base_heading + np.random.normal(0, 15)) % 360
        
        # Generate random passenger count
        max_capacity = 60
        if day_type == "Weekday" and 7 <= hours_offset <= 9:  # Morning rush
            passengers = np.random.randint(20, max_capacity + 1)
        elif day_type == "Weekday" and 16 <= hours_offset <= 18:  # Evening rush
            passengers = np.random.randint(20, max_capacity + 1)
        else:
            passengers = np.random.randint(0, 30)
        
        # Generate random fuel level (0-100%)
        fuel_level = np.random.uniform(20, 100)
        
        # Generate random temperature
        base_temp = 15  # Base temperature in Celsius
        season_variation = 10 * np.sin((days_offset / 90) * 2 * np.pi)  # Seasonal variation
        day_variation = 5 * np.sin((hours_offset / 24) * 2 * np.pi)  # Daily variation
        temperature = base_temp + season_variation + day_variation + np.random.normal(0, 2)
        
        # Generate weather condition
        weather = np.random.choice(weather_conditions, p=weather_probabilities)
        
        # Generate traffic condition
        traffic = np.random.choice(traffic_conditions, p=traffic_probabilities)
        
        # Determine if the bus is delayed
        # Probability of delay increases with bad weather and heavy traffic
        delay_prob = 0.05  # Base probability
        
        if weather in ["Heavy Rain", "Snow", "Fog"]:
            delay_prob += 0.15
        elif weather in ["Light Rain", "Cloudy"]:
            delay_prob += 0.05
        
        if traffic in ["Heavy", "Gridlock"]:
            delay_prob += 0.2
        elif traffic == "Moderate":
            delay_prob += 0.1
        
        is_delayed = np.random.random() < delay_prob
        
        # Generate delay minutes if delayed
        if is_delayed:
            if weather in ["Heavy Rain", "Snow"] and traffic in ["Heavy", "Gridlock"]:
                delay_minutes = np.random.randint(15, 45)
            elif weather in ["Light Rain", "Cloudy"] or traffic in ["Moderate"]:
                delay_minutes = np.random.randint(5, 20)
            else:
                delay_minutes = np.random.randint(3, 10)
        else:
            delay_minutes = 0
        
        # Determine if breakdown occurred (rare event)
        is_breakdown = np.random.random() < 0.005  # 0.5% chance
        
        # Get driver ID
        driver_id = driver_assignments[bus_id]
        
        # Add the record
        data["record_id"].append(str(uuid.uuid4()))
        data["timestamp"].append(timestamp)
        data["bus_id"].append(bus_id)
        data["route_id"].append(route_id)
        data["latitude"].append(round(latitude, 6))
        data["longitude"].append(round(longitude, 6))
        data["speed_kmh"].append(round(speed, 1))
        data["heading"].append(int(heading))
        data["passengers"].append(passengers)
        data["fuel_level"].append(round(fuel_level, 1))
        data["temperature_c"].append(round(temperature, 1))
        data["weather"].append(weather)
        data["traffic"].append(traffic)
        data["day_type"].append(day_type)
        data["is_delayed"].append(is_delayed)
        data["delay_minutes"].append(delay_minutes)
        data["is_breakdown"].append(is_breakdown)
        data["driver_id"].append(driver_id)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df

# Generate the data
df = generate_bus_gps_data(num_records=500000)

# Save to CSV
output_file = "bus_gps_tracking_data_500k.csv"
df.to_csv(output_file, index=False)

print(f"Generated {len(df)} records and saved to {output_file}")

# Display sample of the data
print("\nSample data:")
print(df.head())

# Display summary statistics
print("\nSummary statistics:")
numeric_columns = ["speed_kmh", "passengers", "fuel_level", "temperature_c", "delay_minutes"]
print(df[numeric_columns].describe())

# Count of records by weather condition
print("\nRecords by weather condition:")
print(df["weather"].value_counts())

# Count of records by traffic condition
print("\nRecords by traffic condition:")
print(df["traffic"].value_counts())

# Count of delayed buses
print(f"\nDelayed buses: {df['is_delayed'].sum()} ({df['is_delayed'].mean()*100:.2f}%)")

# Count of breakdowns
print(f"\nBreakdowns: {df['is_breakdown'].sum()} ({df['is_breakdown'].mean()*100:.2f}%)")
