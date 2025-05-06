import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import uuid
import math

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_bus_stops(route_id, start_point, end_point, num_stops):
    """
    Generate a list of bus stops along a route, forming a transit loop.
    
    Parameters:
    -----------
    route_id : str
        Identifier for the route
    start_point : tuple
        (latitude, longitude) of start point
    end_point : tuple
        (latitude, longitude) of end point
    num_stops : int
        Number of stops to generate along the route
    
    Returns:
    --------
    List of dictionaries, each representing a bus stop with:
    - stop_id
    - name
    - latitude
    - longitude
    - avg_dwell_time (seconds)
    """
    stops = []
    
    # Create a loop from start → middle (end point) → back to start
    # First half: from start to end
    for i in range(num_stops // 2):
        progress = i / (num_stops // 2 - 1) if (num_stops // 2 - 1) > 0 else 0
        lat = start_point[0] + progress * (end_point[0] - start_point[0])
        lon = start_point[1] + progress * (end_point[1] - start_point[1])
        
        # Add some noise to make it realistic (not a straight line)
        lat += np.random.normal(0, 0.002)
        lon += np.random.normal(0, 0.002)
        
        stop_id = f"{route_id}_S{str(i+1).zfill(2)}"
        name = f"Stop {i+1}"
        
        # Average dwell time: 30-120 seconds (longer at popular stops)
        popularity = np.random.random()
        if popularity > 0.8:  # Very popular stop
            avg_dwell_time = np.random.randint(60, 120)
        else:
            avg_dwell_time = np.random.randint(30, 60)
        
        stops.append({
            "stop_id": stop_id,
            "name": name,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "avg_dwell_time": avg_dwell_time,
            "is_popular": popularity > 0.8
        })
    
    # Second half: from end back to start (completing the loop)
    for i in range(num_stops // 2):
        progress = i / (num_stops // 2 - 1) if (num_stops // 2 - 1) > 0 else 0
        lat = end_point[0] + progress * (start_point[0] - end_point[0])
        lon = end_point[1] + progress * (start_point[1] - end_point[1])
        
        # Add some noise
        lat += np.random.normal(0, 0.002)
        lon += np.random.normal(0, 0.002)
        
        j = i + num_stops // 2
        stop_id = f"{route_id}_S{str(j+1).zfill(2)}"
        name = f"Stop {j+1}"
        
        # Average dwell time: 30-120 seconds
        popularity = np.random.random()
        if popularity > 0.8:  # Very popular stop
            avg_dwell_time = np.random.randint(60, 120)
        else:
            avg_dwell_time = np.random.randint(30, 60)
        
        stops.append({
            "stop_id": stop_id,
            "name": name,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "avg_dwell_time": avg_dwell_time,
            "is_popular": popularity > 0.8
        })
    
    return stops

def calculate_distance(point1, point2):
    """Calculate distance between two lat/long points in kilometers"""
    # Approximate conversion of lat/long differences to kilometers
    lat_diff = abs(point1[0] - point2[0]) * 111  # 1 degree latitude is approx 111 km
    lon_diff = abs(point1[1] - point2[1]) * 111 * math.cos(math.radians((point1[0] + point2[0]) / 2))
    return math.sqrt(lat_diff**2 + lon_diff**2)

def generate_route_details(route_ids, city_center=(40.7128, -74.0060)):
    """
    Generate detailed route information including stops, detours, and closures
    
    Parameters:
    -----------
    route_ids : list
        List of route identifiers
    city_center : tuple
        (latitude, longitude) of city center
    
    Returns:
    --------
    Dictionary of route details
    """
    route_details = {}
    all_stops = []
    
    # Generate potential closures and detours (for a 30-day period)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 30)
    
    # Generate random closures (5-10 throughout the dataset time period)
    num_closures = random.randint(5, 10)
    closures = []
    for _ in range(num_closures):
        closure_start = start_date + timedelta(days=random.randint(0, 29), 
                                              hours=random.randint(0, 23))
        closure_duration = timedelta(hours=random.randint(1, 24))
        closures.append({
            "start": closure_start,
            "end": closure_start + closure_duration,
            "affected_stops": []  # Will fill in after stops are generated
        })
    
    # Create route details
    for route_id in route_ids:
        # Generate random start and end points within reasonable distance of city center
        start_lat = city_center[0] + np.random.uniform(-0.1, 0.1)
        start_lon = city_center[1] + np.random.uniform(-0.1, 0.1)
        end_lat = city_center[0] + np.random.uniform(-0.1, 0.1)
        end_lon = city_center[1] + np.random.uniform(-0.1, 0.1)
        
        # Ensure start and end points are different enough
        while abs(start_lat - end_lat) < 0.02 and abs(start_lon - end_lon) < 0.02:
            end_lat = city_center[0] + np.random.uniform(-0.1, 0.1)
            end_lon = city_center[1] + np.random.uniform(-0.1, 0.1)
        
        # Number of stops for this route (between 8 and 24, always even for loop)
        num_stops = random.randrange(8, 25, 2)  # Always even number
        
        # Generate stops
        stops = generate_bus_stops(route_id, (start_lat, start_lon), (end_lat, end_lon), num_stops)
        all_stops.extend(stops)
        
        # Calculate total route distance
        total_distance = 0
        for i in range(len(stops) - 1):
            point1 = (stops[i]["latitude"], stops[i]["longitude"])
            point2 = (stops[i+1]["latitude"], stops[i+1]["longitude"])
            total_distance += calculate_distance(point1, point2)
        # Add last segment to complete the loop
        point1 = (stops[-1]["latitude"], stops[-1]["longitude"])
        point2 = (stops[0]["latitude"], stops[0]["longitude"])
        total_distance += calculate_distance(point1, point2)
        
        # Calculate average speed and trip duration
        avg_speed_kmh = np.random.uniform(15, 35)  # Average bus speed in km/h
        avg_trip_duration_minutes = (total_distance / avg_speed_kmh) * 60
        
        # Account for dwell time at stops
        total_dwell_time_minutes = sum([stop["avg_dwell_time"] for stop in stops]) / 60
        avg_trip_duration_minutes += total_dwell_time_minutes
        
        # Randomly assign some stops to be affected by closures
        for closure in closures:
            if random.random() < 0.3:  # 30% chance this route is affected by each closure
                # Pick 1-3 random stops to be affected
                num_affected = random.randint(1, min(3, len(stops)))
                affected_stops = random.sample([stop["stop_id"] for stop in stops], num_affected)
                closure["affected_stops"].extend(affected_stops)
        
        # Define detour routes for certain conditions
        # Detours are alternative paths between stops that may be longer
        detours = []
        num_detours = random.randint(1, 3)
        for _ in range(num_detours):
            # Pick two random sequential stops to create a detour between
            if len(stops) >= 3:
                idx1 = random.randint(0, len(stops) - 2)
                idx2 = idx1 + 1
                
                # Calculate detour increase (10-50% longer)
                normal_distance = calculate_distance(
                    (stops[idx1]["latitude"], stops[idx1]["longitude"]),
                    (stops[idx2]["latitude"], stops[idx2]["longitude"])
                )
                detour_factor = random.uniform(1.1, 1.5)  # 10-50% longer
                detour_distance = normal_distance * detour_factor
                
                detours.append({
                    "from_stop": stops[idx1]["stop_id"],
                    "to_stop": stops[idx2]["stop_id"],
                    "normal_distance_km": normal_distance,
                    "detour_distance_km": detour_distance,
                    "active_dates": []  # Will populate with specific dates when used
                })
        
        route_details[route_id] = {
            "start": (start_lat, start_lon),
            "end": (end_lat, end_lon),
            "stops": stops,
            "avg_speed_kmh": avg_speed_kmh,
            "avg_trip_duration_minutes": avg_trip_duration_minutes,
            "total_distance_km": total_distance,
            "detours": detours
        }
    
    return route_details, closures

def generate_weather_events(start_date, num_days=30):
    """
    Generate weather events that persist for several hours and affect all buses
    
    Parameters:
    -----------
    start_date : datetime
        The starting date for the simulation
    num_days : int
        Number of days to generate events for
    
    Returns:
    --------
    List of weather events, each with:
    - start_time : datetime
    - end_time : datetime
    - weather_type : str
    - severity : str (affects traffic)
    """
    events = []
    
    # Generate 5-15 significant weather events over the time period
    num_events = random.randint(5, 15)
    
    for _ in range(num_events):
        # Random day and time
        event_day = random.randint(0, num_days - 1)
        event_hour = random.randint(0, 23)
        
        # Event start time
        event_start = start_date + timedelta(days=event_day, hours=event_hour)
        
        # Event duration (2-12 hours)
        duration_hours = random.randint(2, 12)
        event_end = event_start + timedelta(hours=duration_hours)
        
        # Weather type - more severe events are less common
        weather_types = {
            "Snow": 0.15,
            "Heavy Rain": 0.25,
            "Fog": 0.15,
            "Light Rain": 0.30,
            "Cloudy": 0.15
        }
        
        weather_type = random.choices(
            list(weather_types.keys()),
            weights=list(weather_types.values()),
            k=1
        )[0]
        
        # Traffic impact based on weather severity
        if weather_type in ["Snow", "Heavy Rain"]:
            severity = random.choice(["Heavy", "Gridlock"])
        elif weather_type in ["Fog", "Light Rain"]:
            severity = random.choice(["Moderate", "Heavy"])
        else:  # Cloudy
            severity = random.choice(["Light", "Moderate"])
        
        events.append({
            "start_time": event_start,
            "end_time": event_end,
            "weather_type": weather_type,
            "severity": severity
        })
    
    return events

def generate_synthetic_bus_data(num_records=100000, num_buses=50, num_routes=10, 
                               city_center=(40.7128, -74.0060)):
    """
    Generate synthetic bus GPS and stop-level data with realistic transit loops
    
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
    Two DataFrames:
    1. Bus GPS tracking data
    2. Stop-level arrival/departure data
    """
    print("Generating synthetic bus data...")
    
    # Create bus IDs
    bus_ids = [f"BUS_{str(i).zfill(4)}" for i in range(1, num_buses + 1)]
    
    # Create route IDs
    route_ids = [f"R{str(i).zfill(2)}" for i in range(1, num_routes + 1)]
    
    # Generate route details with stops, detours, and closures
    route_details, closures = generate_route_details(route_ids, city_center)
    
    # Generate weather events for the time period
    start_date = datetime(2023, 1, 1)
    weather_events = generate_weather_events(start_date)
    print(f"Generated {len(weather_events)} weather events")
    
    # Weather conditions (for non-event periods)
    weather_conditions = ["Clear", "Cloudy", "Light Rain", "Heavy Rain", "Snow", "Fog"]
    weather_probabilities = [0.50, 0.25, 0.10, 0.05, 0.05, 0.05]

    # Traffic conditions (for non-event periods)
    traffic_conditions = ["Light", "Moderate", "Heavy", "Gridlock"]
    traffic_probabilities = [0.4, 0.3, 0.2, 0.1]
    
    # Passenger distribution parameters for different times of day
    passenger_patterns = {
        "night_owl": {  # 12am-5am
            "boarding_range": (0, 5),
            "alighting_range": (0, 3),
            "onboard_range": (0, 10)
        },
        "early_morning": {  # 5am-7am
            "boarding_range": (1, 10),
            "alighting_range": (0, 5),
            "onboard_range": (5, 15)
        },
        "morning_rush": {  # 7am-9am
            "boarding_range": (5, 25),
            "alighting_range": (3, 15),
            "onboard_range": (15, 40)
        },
        "mid_day": {  # 9am-4pm
            "boarding_range": (2, 15),
            "alighting_range": (2, 10),
            "onboard_range": (10, 30)
        },
        "evening_rush": {  # 4pm-7pm
            "boarding_range": (5, 25),
            "alighting_range": (3, 15),
            "onboard_range": (15, 40)
        },
        "evening": {  # 7pm-10pm
            "boarding_range": (1, 10),
            "alighting_range": (1, 8),
            "onboard_range": (5, 20)
        },
        "late_night": {  # 10pm-12am
            "boarding_range": (0, 8),
            "alighting_range": (0, 5),
            "onboard_range": (0, 15)
        }
    }
    
    # Parameters for dwell time calculation
    boarding_time_per_passenger = 2.5  # seconds per passenger boarding
    alighting_time_per_passenger = 1.5  # seconds per passenger alighting
    door_operation_time = 4.0  # seconds to open/close doors
    base_dwell_time = 15.0  # minimum dwell time in seconds
    
    # Bus assignments to routes
    bus_to_route = {}
    for bus_id in bus_ids:
        bus_to_route[bus_id] = random.choice(route_ids)
    
    # Driver assignments
    driver_assignments = {}
    for bus_id in bus_ids:
        driver_assignments[bus_id] = f"D{str(random.randint(1000, 9999))}"
    
    # Date range
    start_date = datetime(2023, 1, 1)
    
    # Initialize data dictionaries
    gps_data = {
        "record_id": [],
        "timestamp": [],
        "bus_id": [],
        "route_id": [],
        "stop_id": [],  # Current/next stop
        "latitude": [],
        "longitude": [],
        "speed_kmh": [],
        "heading": [],
        "distance_to_next_stop_km": [],
        "estimated_arrival_time": [],
        "actual_arrival_time": [],  # Will be filled for arrival events
        "passengers_on_board": [],
        "fuel_level": [],
        "temperature_c": [],
        "weather": [],
        "traffic": [],
        "day_type": [],
        "is_delayed": [],
        "delay_minutes": [],
        "is_detour_active": [],
        "is_stop_closed": [],
        "is_breakdown": [],
        "driver_id": []
    }
    
    stop_data = {
        "record_id": [],
        "timestamp": [],
        "bus_id": [],
        "route_id": [],
        "stop_id": [],
        "stop_name": [],
        "arrival_time": [],
        "departure_time": [],
        "dwell_time_seconds": [],
        "passengers_boarding": [],
        "passengers_alighting": [],
        "scheduled_arrival_time": [],
        "delay_minutes": [],
        "is_stop_closed": [],
        "weather": [],
        "traffic": []
    }
    
    # Helper function to determine time of day category
    def get_time_of_day(hour):
        if 0 <= hour < 5:
            return "night_owl"
        elif 5 <= hour < 7:
            return "early_morning"
        elif 7 <= hour < 9:
            return "morning_rush"
        elif 9 <= hour < 16:
            return "mid_day"
        elif 16 <= hour < 19:
            return "evening_rush"
        elif 19 <= hour < 22:
            return "evening"
        else:  # 22-24
            return "late_night"
    
    # Generate trip simulations to create realistic sequential data
    print("Generating trips...")
    trips_processed = 0
    records_generated = 0
    
    while records_generated < num_records:
        # Select a random bus and its route
        bus_id = random.choice(bus_ids)
        route_id = bus_to_route[bus_id]
        route = route_details[route_id]
        
        # Generate a random start time for this trip (buses run 24/7 now)
        days_offset = np.random.randint(0, 30)
        hours_offset = np.random.randint(0, 24)  # Full 24-hour coverage
        minutes_offset = np.random.randint(0, 60)
        trip_start_time = start_date + timedelta(days=days_offset, hours=hours_offset, 
                                               minutes=minutes_offset)
        
        # Get time of day category for passenger patterns
        time_of_day = get_time_of_day(hours_offset)
        
        # Determine day type
        if trip_start_time.weekday() < 5:  # Monday=0, Sunday=6
            day_type = "Weekday"
        else:
            day_type = "Weekend"
        
        # Check if trip occurs during a weather event
        active_weather_event = None
        for event in weather_events:
            if event["start_time"] <= trip_start_time <= event["end_time"]:
                active_weather_event = event
                break
        
        # Select weather and traffic based on events or random conditions
        if active_weather_event:
            weather = active_weather_event["weather_type"]
            traffic = active_weather_event["severity"]
        else:
            # Normal weather and traffic
            weather = np.random.choice(weather_conditions, p=weather_probabilities)
            traffic = np.random.choice(traffic_conditions, p=traffic_probabilities)
        
        # Temperature
        base_temp = 15  # Base temperature in Celsius
        season_variation = 10 * np.sin((days_offset / 90) * 2 * np.pi)  # Seasonal variation
        day_variation = 5 * np.sin((hours_offset / 24) * 2 * np.pi)  # Daily variation
        
        # Adjust temperature based on weather
        if weather == "Snow":
            temp_adjustment = random.uniform(-15, -5)
        elif weather == "Heavy Rain":
            temp_adjustment = random.uniform(-5, 0)
        elif weather == "Fog":
            temp_adjustment = random.uniform(-3, 2)
        elif weather == "Light Rain":
            temp_adjustment = random.uniform(-2, 3)
        else:
            temp_adjustment = random.uniform(-2, 5)
            
        temperature = base_temp + season_variation + day_variation + temp_adjustment
        
        # Random starting fuel level (50-100%)
        fuel_level = np.random.uniform(50, 100)
        
        # Determine if any detours are active for this trip
        active_detours = []
        for detour in route["detours"]:
            if random.random() < 0.1:  # 10% chance a detour is active
                detour["active_dates"].append(trip_start_time.date())
                active_detours.append(detour)
        
        # Get driver ID
        driver_id = driver_assignments[bus_id]
        
        # Initialize passengers (based on time of day)
        pattern = passenger_patterns[time_of_day]
        passengers = np.random.randint(pattern["onboard_range"][0], pattern["onboard_range"][1] + 1)
        
        # Adjust passenger count for weekend
        if day_type == "Weekend":
            passengers = max(0, int(passengers * 0.7))  # 30% fewer passengers on weekends
        
        # Calculate base delay probability
        delay_prob = 0.05  # Base probability
        
        if weather in ["Heavy Rain", "Snow", "Fog"]:
            delay_prob += 0.15
        elif weather in ["Light Rain", "Cloudy"]:
            delay_prob += 0.05
        
        if traffic in ["Heavy", "Gridlock"]:
            delay_prob += 0.2
        elif traffic == "Moderate":
            delay_prob += 0.1
            
        # Night-time operations have slightly higher delay risk due to reduced staff
        if time_of_day in ["night_owl", "late_night"]:
            delay_prob += 0.03
            
        # Determine if this trip is delayed
        is_delayed = np.random.random() < delay_prob
        
        # Delay minutes for the trip
        if is_delayed:
            if weather in ["Heavy Rain", "Snow"] and traffic in ["Heavy", "Gridlock"]:
                base_delay = np.random.randint(15, 30)
            elif weather in ["Light Rain", "Cloudy"] or traffic in ["Moderate"]:
                base_delay = np.random.randint(5, 15)
            else:
                base_delay = np.random.randint(3, 10)
        else:
            base_delay = 0
            
        # Check for breakdowns (rare)
        is_breakdown = np.random.random() < 0.005  # 0.5% chance
        if is_breakdown:
            base_delay += np.random.randint(20, 60)  # Add 20-60 minutes for breakdown
        
        # Simulate the journey through all stops
        current_time = trip_start_time
        stops = route["stops"]
        
        for i in range(len(stops)):
            current_stop = stops[i]
            next_stop_idx = (i + 1) % len(stops)  # Circular reference for loop routes
            next_stop = stops[next_stop_idx]
            
            # Check if stop is closed due to any closure
            is_stop_closed = False
            for closure in closures:
                if (current_stop["stop_id"] in closure["affected_stops"] and 
                    closure["start"] <= current_time <= closure["end"]):
                    is_stop_closed = True
                    break
            
            # If stop is closed, skip it but add time for detour
            if is_stop_closed:
                # Still record the stop as closed in the data
                stop_record_id = str(uuid.uuid4())
                stop_data["record_id"].append(stop_record_id)
                stop_data["timestamp"].append(current_time)
                stop_data["bus_id"].append(bus_id)
                stop_data["route_id"].append(route_id)
                stop_data["stop_id"].append(current_stop["stop_id"])
                stop_data["stop_name"].append(current_stop["name"])
                stop_data["arrival_time"].append(None)  # Bus didn't actually arrive
                stop_data["departure_time"].append(None)
                stop_data["dwell_time_seconds"].append(0)
                stop_data["passengers_boarding"].append(0)
                stop_data["passengers_alighting"].append(0)
                stop_data["scheduled_arrival_time"].append(current_time)
                stop_data["delay_minutes"].append(base_delay)
                stop_data["is_stop_closed"].append(True)
                stop_data["weather"].append(weather)
                stop_data["traffic"].append(traffic)
                
                # Add extra time for bypassing closed stop
                current_time += timedelta(minutes=random.randint(2, 5))
                continue
            
            # Check if there's a detour between this stop and the next
            is_detour_active = False
            detour_extra_distance = 0
            for detour in active_detours:
                if (detour["from_stop"] == current_stop["stop_id"] and 
                    detour["to_stop"] == next_stop["stop_id"]):
                    is_detour_active = True
                    detour_extra_distance = detour["detour_distance_km"] - detour["normal_distance_km"]
                    break
            
            # Calculate distance to next stop
            point1 = (current_stop["latitude"], current_stop["longitude"])
            point2 = (next_stop["latitude"], next_stop["longitude"])
            distance = calculate_distance(point1, point2)
            
            if is_detour_active:
                distance += detour_extra_distance
            
            # Calculate travel time to next stop based on distance, speed, and conditions
            speed_adjustment = 1.0
            if traffic == "Light":
                speed_adjustment = 1.1
            elif traffic == "Moderate":
                speed_adjustment = 0.9
            elif traffic == "Heavy":
                speed_adjustment = 0.7
            elif traffic == "Gridlock":
                speed_adjustment = 0.4
                
            if weather in ["Heavy Rain", "Snow", "Fog"]:
                speed_adjustment *= 0.8
            elif weather in ["Light Rain"]:
                speed_adjustment *= 0.9
                
            # Night-time operations tend to be faster due to less traffic
            if time_of_day in ["night_owl", "late_night"]:
                speed_adjustment *= 1.2
                
            actual_speed = route["avg_speed_kmh"] * speed_adjustment
            travel_time_hours = distance / actual_speed
            travel_time_minutes = travel_time_hours * 60
            
            # Add random variation (±10%)
            travel_time_minutes *= random.uniform(0.9, 1.1)
            
            # Add delay if this trip is delayed
            travel_time_minutes += base_delay * (travel_time_minutes / route["avg_trip_duration_minutes"])
            
            # Calculate arrival time at next stop
            scheduled_arrival = current_time + timedelta(minutes=travel_time_minutes)
            actual_arrival = scheduled_arrival
            
            # Calculate passengers boarding and alighting based on time of day
            pattern = passenger_patterns[get_time_of_day(actual_arrival.hour)]
            
            # More people at popular stops
            if current_stop["is_popular"]:
                boarding_multiplier = 1.5
                alighting_multiplier = 1.3
            else:
                boarding_multiplier = 1.0
                alighting_multiplier = 1.0
                
            # Calculate passengers alighting - limited by how many are on the bus
            max_alighting = min(passengers, pattern["alighting_range"][1])
            passengers_alighting = min(
                random.randint(pattern["alighting_range"][0], max(pattern["alighting_range"][0], max_alighting)),
                passengers
            )
            passengers_alighting = int(passengers_alighting * alighting_multiplier)
            
            # Calculate passengers boarding
            passengers_boarding = random.randint(
                pattern["boarding_range"][0], 
                pattern["boarding_range"][1]
            )
            passengers_boarding = int(passengers_boarding * boarding_multiplier)
            
            # Weekend adjustment
            if day_type == "Weekend":
                passengers_boarding = max(0, int(passengers_boarding * 0.7))
                passengers_alighting = max(0, int(passengers_alighting * 0.7))
                
            # Update passenger count
            passengers = passengers - passengers_alighting + passengers_boarding
            
            # Calculate dwell time based on passenger activity
            # Formula: base_time + (boarding_time * boarding_passengers) + (alighting_time * alighting_passengers)
            boarding_time = boarding_time_per_passenger * passengers_boarding
            alighting_time = alighting_time_per_passenger * passengers_alighting
            
            # Only count the longer of boarding/alighting if they happen simultaneously at different doors
            passenger_time = max(boarding_time, alighting_time)
            
            # Only add door operation time if passengers are boarding/alighting
            if passengers_boarding > 0 or passengers_alighting > 0:
                actual_dwell_time = base_dwell_time + passenger_time + door_operation_time
            else:
                actual_dwell_time = base_dwell_time
                
            # Cap at a reasonable maximum (5 minutes)
            actual_dwell_time = min(actual_dwell_time, 300)
            
            # Calculate departure time
            departure_time = actual_arrival + timedelta(seconds=actual_dwell_time)
            
            # Update current time for next iteration
            current_time = departure_time
            
            # Create stop-level record
            stop_record_id = str(uuid.uuid4())
            stop_data["record_id"].append(stop_record_id)
            stop_data["timestamp"].append(actual_arrival)
            stop_data["bus_id"].append(bus_id)
            stop_data["route_id"].append(route_id)
            stop_data["stop_id"].append(current_stop["stop_id"])
            stop_data["stop_name"].append(current_stop["name"])
            stop_data["arrival_time"].append(actual_arrival)
            stop_data["departure_time"].append(departure_time)
            stop_data["dwell_time_seconds"].append(actual_dwell_time)
            stop_data["passengers_boarding"].append(passengers_boarding)
            stop_data["passengers_alighting"].append(passengers_alighting)
            stop_data["scheduled_arrival_time"].append(scheduled_arrival)
            stop_data["delay_minutes"].append(base_delay)
            stop_data["is_stop_closed"].append(False)
            stop_data["weather"].append(weather)
            stop_data["traffic"].append(traffic)
            
            # Add departure_time to current_stop for use in GPS point calculations
            current_stop["departure_time"] = departure_time
            
            # Generate GPS points for the journey between stops
            # Number of GPS points depends on distance and time
            num_gps_points = max(int(travel_time_minutes / 2), 1)  # Approx every 2 minutes
            
            for j in range(num_gps_points):
                # Skip some GPS points if we've reached our record limit
                if records_generated >= num_records:
                    break
                
                # Calculate position along the route from current to next stop
                progress = j / num_gps_points
                
                # If detour is active, add some randomness to the path
                if is_detour_active and j < num_gps_points - 1:
                    # More winding path for detour
                    lat_jitter = np.random.normal(0, 0.005)
                    lon_jitter = np.random.normal(0, 0.005)
                else:
                    lat_jitter = np.random.normal(0, 0.0005)
                    lon_jitter = np.random.normal(0, 0.0005)
                
                lat = current_stop["latitude"] + progress * (next_stop["latitude"] - current_stop["latitude"]) + lat_jitter
                lon = current_stop["longitude"] + progress * (next_stop["longitude"] - current_stop["longitude"]) + lon_jitter
                
                # Calculate speed at this point
                if j == 0 or j == num_gps_points - 1:
                    # Slower speed near stops
                    speed = actual_speed * random.uniform(0.3, 0.7)
                else:
                    # Full speed in the middle of journey
                    speed = actual_speed * random.uniform(0.8, 1.2)
                    
                    # Slowdowns due to traffic or weather
                    if traffic in ["Heavy", "Gridlock"]:
                        if random.random() < 0.3:
                            speed *= random.uniform(0.3, 0.6)
                    
                # Calculate heading
                if next_stop["latitude"] > current_stop["latitude"]:
                    base_heading = 45 if next_stop["longitude"] > current_stop["longitude"] else 135
                else:
                    base_heading = 315 if next_stop["longitude"] > current_stop["longitude"] else 225
                
                heading = (base_heading + np.random.normal(0, 15)) % 360
                
                # Calculate timestamp for this GPS point
                point_time = current_stop["departure_time"] + timedelta(minutes=travel_time_minutes * progress)
                
                # Calculate distance to next stop
                remaining_distance = distance * (1 - progress)
                
                # Calculate estimated arrival based on current conditions
                remaining_time_hours = remaining_distance / speed
                estimated_arrival = point_time + timedelta(hours=remaining_time_hours)
                
                # Create GPS record
                gps_record_id = str(uuid.uuid4())
                gps_data["record_id"].append(gps_record_id)
                gps_data["timestamp"].append(point_time)
                gps_data["bus_id"].append(bus_id)
                gps_data["route_id"].append(route_id)
                gps_data["stop_id"].append(next_stop["stop_id"])  # Next stop the bus is approaching
                gps_data["latitude"].append(round(lat, 6))
                gps_data["longitude"].append(round(lon, 6))
                gps_data["speed_kmh"].append(round(speed, 1))
                gps_data["heading"].append(int(heading))
                gps_data["distance_to_next_stop_km"].append(round(remaining_distance, 3))
                gps_data["estimated_arrival_time"].append(estimated_arrival)
                gps_data["actual_arrival_time"].append(None)  # Will be filled only for arrival events
                gps_data["passengers_on_board"].append(passengers)
                gps_data["fuel_level"].append(round(fuel_level, 1))
                gps_data["temperature_c"].append(round(temperature, 1))
                gps_data["weather"].append(weather)
                gps_data["traffic"].append(traffic)
                gps_data["day_type"].append(day_type)
                gps_data["is_delayed"].append(is_delayed)
                gps_data["delay_minutes"].append(base_delay)
                gps_data["is_detour_active"].append(is_detour_active)
                gps_data["is_stop_closed"].append(is_stop_closed)
                gps_data["is_breakdown"].append(is_breakdown)
                gps_data["driver_id"].append(driver_id)
                
                records_generated += 1
                
                # Create a special GPS record for the arrival at the stop for the last GPS point
                if j == num_gps_points - 1:
                    gps_record_id = str(uuid.uuid4())
                    gps_data["record_id"].append(gps_record_id)
                    gps_data["timestamp"].append(actual_arrival)
                    gps_data["bus_id"].append(bus_id)
                    gps_data["route_id"].append(route_id)
                    gps_data["stop_id"].append(next_stop["stop_id"])
                    gps_data["latitude"].append(round(next_stop["latitude"], 6))
                    gps_data["longitude"].append(round(next_stop["longitude"], 6))
                    gps_data["speed_kmh"].append(0.0)  # Stopped at the stop
                    gps_data["heading"].append(int(heading))
                    gps_data["distance_to_next_stop_km"].append(0.0)
                    gps_data["estimated_arrival_time"].append(actual_arrival)
                    gps_data["actual_arrival_time"].append(actual_arrival)  # This is an arrival event
                    gps_data["passengers_on_board"].append(passengers)
                    gps_data["fuel_level"].append(round(fuel_level, 1))
                    gps_data["temperature_c"].append(round(temperature, 1))
                    gps_data["weather"].append(weather)
                    gps_data["traffic"].append(traffic)
                    gps_data["day_type"].append(day_type)
                    gps_data["is_delayed"].append(is_delayed)
                    gps_data["delay_minutes"].append(base_delay)
                    gps_data["is_detour_active"].append(is_detour_active)
                    gps_data["is_stop_closed"].append(is_stop_closed)
                    gps_data["is_breakdown"].append(is_breakdown)
                    gps_data["driver_id"].append(driver_id)
                    
                    records_generated += 1
            
            # Small chance to decrease fuel level
            fuel_level -= random.uniform(0.1, 0.5)
            if fuel_level < 20:
                fuel_level = 100  # Refueled
        
        trips_processed += 1
        if trips_processed % 10 == 0:
            print(f"Generated {records_generated}/{num_records} records from {trips_processed} trips")
            
    # Convert to DataFrames
    gps_df = pd.DataFrame(gps_data)
    stop_df = pd.DataFrame(stop_data)
    
    # Sort by timestamp
    gps_df = gps_df.sort_values("timestamp")
    stop_df = stop_df.sort_values("timestamp")
    
    # Take only the number of records requested
    gps_df = gps_df.head(num_records)
    
    return gps_df, stop_df


# Generate the data
num_records = 100000  # Adjust as needed
gps_data, stop_data = generate_synthetic_bus_data(num_records=num_records, num_buses=50, num_routes=10)

# Save to CSV
gps_output_file = "bus_gps_tracking_data.csv"
stop_output_file = "bus_stop_level_data.csv"
gps_data.to_csv(gps_output_file, index=False)
stop_data.to_csv(stop_output_file, index=False)

print(f"Generated {len(gps_data)} GPS records and saved to {gps_output_file}")
print(f"Generated {len(stop_data)} stop-level records and saved to {stop_output_file}")

# Display sample of the data
print("\nSample GPS data:")
print(gps_data.head())

print("\nSample stop-level data:")
print(stop_data.head())

# Display summary statistics
print("\nGPS data summary statistics:")
numeric_columns = ["speed_kmh", "passengers_on_board", "fuel_level", "temperature_c", "delay_minutes"]
print(gps_data[numeric_columns].describe())

# Count of records by weather condition
print("\nRecords by weather condition:")
print(gps_data["weather"].value_counts())

# Count of records by traffic condition
print("\nRecords by traffic condition:")
print(gps_data["traffic"].value_counts())

# Count of delayed buses
print(f"\nDelayed buses: {gps_data['is_delayed'].sum()} ({gps_data['is_delayed'].mean()*100:.2f}%)")

# Count of detours
print(f"\nRecords with active detours: {gps_data['is_detour_active'].sum()} ({gps_data['is_detour_active'].mean()*100:.2f}%)")

# Count of closed stops
print(f"\nClosed stop encounters: {gps_data['is_stop_closed'].sum()} ({gps_data['is_stop_closed'].mean()*100:.2f}%)")

# Count of breakdowns
print(f"\nBreakdowns: {gps_data['is_breakdown'].sum()} ({gps_data['is_breakdown'].mean()*100:.2f}%)")
