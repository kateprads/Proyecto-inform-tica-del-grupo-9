class NavAirport:
   
    
    def __init__(self, name, sids=None, stars=None):
  
        self.name = name
        self.sids = sids if sids is not None else []
        self.stars = stars if stars is not None else []
    
    def __str__(self):
      
        return f"Airport: {self.name} (SIDs: {len(self.sids)}, STARs: {len(self.stars)})"
    
    def get_info(self):
     
        info = f"Airport: {self.name}\n"
        
        info += "SIDs (Departure points):\n"
        for sid in self.sids:
            info += f"  {sid}\n"
        
        info += "STARs (Arrival points):\n"
        for star in self.stars:
            info += f"  {star}\n"
        
        return info
    
    def add_sid(self, navpoint_id):
        
        if navpoint_id not in self.sids:
            self.sids.append(navpoint_id)
    
    def add_star(self, navpoint_id):
       
        if navpoint_id not in self.stars:
            self.stars.append(navpoint_id)
    
    def get_first_sid(self):
       
        return self.sids[0] if self.sids else None

# Functions for file handling and airport operations

def read_airports_from_file(filename, navpoints):
   
    airports = {}
    current_airport = None
    
    try:
        with open(filename, 'r') as file:
            for line in file:
                # Skip header or empty lines
                if line.strip() == '' or line.startswith('....'):
                    continue
                
                parts = line.strip().split()
                if not parts:
                    continue
                
                # If this line contains an airport name (in bold according to the example)
                if len(parts) == 1 or (len(parts) > 1 and parts[1].startswith('.') and not parts[0].startswith('.')):
                    airport_name = parts[0]
                    current_airport = NavAirport(airport_name)
                    airports[airport_name] = current_airport
                # If we have an airport and this line contains SIDs or STARs
                elif current_airport is not None and len(parts) >= 2:
                    nav_id = None
                    try:
                        # Try to find the navigation point ID by matching its name from the line
                        for np_id, np in navpoints.items():
                            if np.name == parts[0]:
                                nav_id = np_id
                                break
                        
                        # If we found a navigation point ID
                        if nav_id is not None:
                            # Check if it's a SID or STAR
                            if "SID" in line or "departure" in line.lower():
                                current_airport.add_sid(nav_id)
                            elif "STAR" in line or "arrival" in line.lower():
                                current_airport.add_star(nav_id)
                    except (ValueError, IndexError):
                        print(f"Warning: Could not parse line for airport {current_airport.name}: {line}")
    
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
    except Exception as e:
        print(f"Error reading file: {e}")
    
    return airports

def get_airport_by_name(airports, name):
  
    return airports.get(name)

def parse_airport_file(filename):
   
    airports = {}
    
    try:
        with open(filename, 'r') as file:
            # Skip header lines until we find "......"
            header_passed = False
            current_airport = None
            
            for line in file:
                if not header_passed:
                    if line.strip() == "......":
                        header_passed = True
                    continue
                
                # Skip empty lines or separator lines
                if line.strip() == '' or line.strip() == "......" or line.startswith('....'):
                    continue
                
                parts = line.strip().split()
                if not parts:
                    continue
                
                # Check if this is an airport line (4-letter code)
                if len(parts[0]) == 4 and parts[0].isupper():
                    airport_name = parts[0]
                    current_airport = NavAirport(airport_name)
                    airports[airport_name] = current_airport
                
                # If we have an airport and this might be a SID or STAR
                elif current_airport is not None and len(parts) >= 1:
                    try:
                        # Try to parse the first part as a navigation point ID
                        try:
                            navpoint_id = int(parts[0])
                        except ValueError:
                            # If not a number, check other parts
                            navpoint_id = None
                            for part in parts:
                                try:
                                    navpoint_id = int(part)
                                    break
                                except ValueError:
                                    continue
                        
                        if navpoint_id is not None:
                            # Check if it's a SID or STAR
                            line_lower = line.lower()
                            if "sid" in line_lower or "departure" in line_lower:
                                current_airport.add_sid(navpoint_id)
                            elif "star" in line_lower or "arrival" in line_lower:
                                current_airport.add_star(navpoint_id)
                    except Exception as e:
                        print(f"Warning: Error parsing line for airport {current_airport.name}: {line} - {e}")
    
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
    except Exception as e:
        print(f"Error parsing airport file: {e}")
    
    return airports
   

if __name__ == "__main__":
    # Test the NavAirport class
    test_airport = NavAirport("LEIB", [6063], [6061])
    print(test_airport)
    print(test_airport.get_info())
    
    # Test adding SIDs and STARs
    test_airport.add_sid(6062)
    test_airport.add_star(6063)
    print("After adding more navigation points:")
    print(test_airport.get_info())
    
    # Test reading from file (if file exists)
    try:
        # First, attempt to use the custom parser
        test_airports = parse_airport_file("Cat_aer.txt")
        print(f"Read {len(test_airports)} airports from file using custom parser.")
        
        # Check if LEIB (Ibiza) is in the airports
        ibiza = get_airport_by_name(test_airports, "LEIB")
        if ibiza:
            print("Found Ibiza airport:")
            print(ibiza.get_info())
        else:
            print("Ibiza airport not found in the parsed data.")
    except Exception as e:
        print(f"Test error: {e}")