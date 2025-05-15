class NavPoint:
    """
    Class that represents a navigation point in the airspace.
    
    Attributes:
        number (int): The node number in the nav.txt file
        name (str): Name of the navigation point
        latitude (float): Geographical latitude in degrees
        longitude (float): Geographical longitude in degrees
    """
    
    def __init__(self, number, name, latitude, longitude):
        """
        Constructor for the NavPoint class.
        
        Args:
            number (int): The node number in the nav.txt file
            name (str): Name of the navigation point
            latitude (float): Geographical latitude in degrees
            longitude (float): Geographical longitude in degrees
        """
        self.number = number
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
    
    def __str__(self):
        
        return f"{self.number} {self.name} ({self.latitude}, {self.longitude})"
    
    def get_info(self):
        
        return f"Navigation Point #{self.number}\nName: {self.name}\nCoordinates: ({self.latitude}, {self.longitude})"

# Functions for file handling and navigation point operations

def read_navpoints_from_file(filename):
    """
    Reads navigation points from a file.
    
    """
    navpoints = {}
    
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
                if len(parts) >= 4:
                    try:
                        number = int(parts[0])
                        name = parts[1]
                        latitude = float(parts[2])
                        longitude = float(parts[3])
                        
                        navpoints[number] = NavPoint(number, name, latitude, longitude)
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse line: {line} - Error: {e}")
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
    except Exception as e:
        print(f"Error reading file: {e}")
    
    return navpoints

def get_navpoint_by_id(navpoints, navpoint_id):
    
    return navpoints.get(navpoint_id)

def calculate_distance(navpoint1, navpoint2):
    """
    Calculate the geographical distance between two navigation points.
    
    """
    from math import radians, cos, sin, asin, sqrt
    
    # Convert latitude and longitude from degrees to radians
    lat1 = radians(navpoint1.latitude)
    lon1 = radians(navpoint1.longitude)
    lat2 = radians(navpoint2.latitude)
    lon2 = radians(navpoint2.longitude)
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in kilometers is 6371
    km = 6371 * c
    return km

if __name__ == "__main__":
    # Test the NavPoint class
    test_navpoint = NavPoint(6063, "IZA.D", 38.8731546833, 1.37242975)
    print(test_navpoint)
    print(test_navpoint.get_info())
    
    # Test reading from file (if file exists)
    try:
        test_navpoints = read_navpoints_from_file("Cat_nav.txt")
        print(f"Read {len(test_navpoints)} navigation points from file.")
        
        # Test distance calculation if at least 2 points exist
        if len(test_navpoints) >= 2:
            keys = list(test_navpoints.keys())
            np1 = test_navpoints[keys[0]]
            np2 = test_navpoints[keys[1]]
            distance = calculate_distance(np1, np2)
            print(f"Distance between {np1.name} and {np2.name}: {distance:.2f} km")
    except Exception as e:
        print(f"Test error: {e}")