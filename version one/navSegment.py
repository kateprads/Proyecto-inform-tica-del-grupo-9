class NavSegment:
    """
    Class that represents a navigation segment connecting two navigation points.
    
    Attributes:
        origin_number (int): The origin node number of the segment
        destination_number (int): The destination node number of the segment
        distance (float): Distance in kilometers to go from origin to destination
    """
    
    def __init__(self, origin_number, destination_number, distance):
        
        self.origin_number = origin_number
        self.destination_number = destination_number
        self.distance = distance
    
    def __str__(self):
     
        return f"Segment: {self.origin_number} -> {self.destination_number} (Distance: {self.distance} km)"
    
    def get_info(self):
        
        return f"Segment from {self.origin_number} to {self.destination_number}\nDistance: {self.distance} km"

# Functions for file handling and segment operations

def read_segments_from_file(filename):
   
    segments = []
    
    try:
        with open(filename, 'r') as file:
            # Skip header lines until we find "......"
            header_passed = False
            for line in file:
                if not header_passed:
                    if line.strip() == "......":
                        header_passed = True
                    continue
                
                # Skip empty lines or separator lines
                if line.strip() == '' or line.strip() == "......" or line.startswith('....'):
                    continue
                
                # Parse the line
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        origin = int(parts[0])
                        destination = int(parts[1])
                        distance = float(parts[2])
                        
                        segments.append(NavSegment(origin, destination, distance))
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse line: {line} - Error: {e}")
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
    except Exception as e:
        print(f"Error reading file: {e}")
    
    return segments

def get_segments_by_origin(segments, origin_id):
  
    return [segment for segment in segments if segment.origin_number == origin_id]

def get_segments_by_destination(segments, destination_id):
   
    return [segment for segment in segments if segment.destination_number == destination_id]

def get_segment(segments, origin_id, destination_id):
   
    for segment in segments:
        if segment.origin_number == origin_id and segment.destination_number == destination_id:
            return segment
    return None

if __name__ == "__main__":
    # Test the NavSegment class
    test_segment = NavSegment(6063, 6937, 48.55701)
    print(test_segment)
    print(test_segment.get_info())
    
    # Test reading from file (if file exists)
    try:
        test_segments = read_segments_from_file("Cat_seg.txt")
        print(f"Read {len(test_segments)} navigation segments from file.")
        
        # Test finding segments by origin
        origin_id = 6063  # IZA.D from the example in the image
        origin_segments = get_segments_by_origin(test_segments, origin_id)
        print(f"Found {len(origin_segments)} segments with origin {origin_id}")
        for segment in origin_segments:
            print(f"  {segment}")
    except Exception as e:
        print(f"Test error: {e}")