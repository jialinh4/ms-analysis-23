from config import activity_threshold
from numpy.linalg import norm

def segment_data(data):
    segments = []
    segment_start = 0
    for i in range(1, len(data)):
        current_magnitude = norm([data['x'].iloc[i], data['y'].iloc[i], data['z'].iloc[i]])
        previous_magnitude = norm([data['x'].iloc[i-1], data['y'].iloc[i-1], data['z'].iloc[i-1]])
        delta = abs(current_magnitude - previous_magnitude)
        
        if delta > activity_threshold:
            segments.append(data.iloc[segment_start:i])
            segment_start = i
            
    # Appending the last segment
    segments.append(data.iloc[segment_start:])
    return segments
