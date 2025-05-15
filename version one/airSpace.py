from navPoint import NavPoint, read_navpoints_from_file
from navSegment import NavSegment, read_segments_from_file
from navAirport import NavAirport, parse_airport_file

class AirSpace:
    """
    Class that represents an airspace with its navigation points, segments, and airports.
    
    Attributes:
        navpoints (dict): Dictionary of NavPoint objects
        navsegments (list): List of NavSegment objects
        navairports (dict): Dictionary of NavAirport objects
        name (str): Name of the airspace
    """
    
    def __init__(self, name="Undefined"):
       
        self.navpoints = {}
        self.navsegments = []
        self.navairports = {}
        self.name = name
    
    def __str__(self):
       
        return f"AirSpace: {self.name} (Points: {len(self.navpoints)}, Segments: {len(self.navsegments)}, Airports: {len(self.navairports)})"
    
    def get_info(self):
     
        info = f"AirSpace: {self.name}\n"
        info += f"Navigation Points: {len(self.navpoints)}\n"
        info += f"Navigation Segments: {len(self.navsegments)}\n"
        info += f"Airports: {len(self.navairports)}\n"
        
        return info
    
    def load_from_files(self, nav_file, seg_file, aer_file):
       
        try:
            # Load navigation points
            self.navpoints = read_navpoints_from_file(nav_file)
            
            # Load segments
            all_segments = read_segments_from_file(seg_file)
            
            # Filter out segments that reference non-existent navigation points
            valid_segments = []
            for segment in all_segments:
                if segment.origin_number in self.navpoints and segment.destination_number in self.navpoints:
                    valid_segments.append(segment)
                else:
                    print(f"Warning: Segment with origin {segment.origin_number} and destination {segment.destination_number} references non-existent navigation point(s).")
            
            self.navsegments = valid_segments
            
            # Load airports
            self.navairports = parse_airport_file(aer_file)
            
            # Validate airport SIDs and STARs
            for airport_name, airport in self.navairports.items():
                airport.sids = [sid for sid in airport.sids if sid in self.navpoints]
                airport.stars = [star for star in airport.stars if star in self.navpoints]
                
                if not airport.sids and not airport.stars:
                    print(f"Warning: Airport {airport_name} has no valid SIDs or STARs.")
            
            return True
        except Exception as e:
            print(f"Error loading airspace data: {e}")
            return False
    
    def get_navpoint(self, navpoint_id):
       
        return self.navpoints.get(navpoint_id)
    
    def get_airport(self, airport_name):
      
        return self.navairports.get(airport_name)
    
    def get_neighbors(self, navpoint_id):
       
        neighbors = []
        
        for segment in self.navsegments:
            if segment.origin_number == navpoint_id:
                destination = self.get_navpoint(segment.destination_number)
                if destination:
                    neighbors.append(destination)
            # Also check if the segment has the point as destination
            # This ensures we get all connected points, not just outgoing connections
            elif segment.destination_number == navpoint_id:
                origin = self.get_navpoint(segment.origin_number)
                if origin:
                    neighbors.append(origin)
        
        return neighbors
    
    def get_segment(self, origin_id, destination_id):
     
        for segment in self.navsegments:
            if segment.origin_number == origin_id and segment.destination_number == destination_id:
                return segment
        return None
    
    def find_shortest_path(self, start_id, end_id):
        
        import heapq
        
        # Check if start and end points exist
        if start_id not in self.navpoints or end_id not in self.navpoints:
            return [], float('inf')
        
        # Initialize distances and previous nodes
        distances = {node_id: float('inf') for node_id in self.navpoints}
        distances[start_id] = 0
        previous = {node_id: None for node_id in self.navpoints}
        
        # Priority queue for Dijkstra's algorithm
        queue = [(0, start_id)]
        
        while queue:
            current_distance, current_id = heapq.heappop(queue)
            
            # If we reached the end, we can terminate
            if current_id == end_id:
                break
            
            # If we have a worse distance than already found, skip
            if current_distance > distances[current_id]:
                continue
            
            # Check all segments that have current_id as origin
            for segment in self.navsegments:
                # Check outgoing connections
                if segment.origin_number == current_id and segment.destination_number in self.navpoints:
                    neighbor_id = segment.destination_number
                    # Calculate distance through current node
                    distance = current_distance + segment.distance
                    
                    # If we found a better path
                    if distance < distances.get(neighbor_id, float('inf')):
                        distances[neighbor_id] = distance
                        previous[neighbor_id] = current_id
                        heapq.heappush(queue, (distance, neighbor_id))
                
                # Also check incoming connections - assuming the graph is bidirectional
                elif segment.destination_number == current_id and segment.origin_number in self.navpoints:
                    neighbor_id = segment.origin_number
                    # Calculate distance through current node
                    distance = current_distance + segment.distance
                    
                    # If we found a better path
                    if distance < distances.get(neighbor_id, float('inf')):
                        distances[neighbor_id] = distance
                        previous[neighbor_id] = current_id
                        heapq.heappush(queue, (distance, neighbor_id))
        
        # Reconstruct the path
        path = []
        current = end_id
        
        while current:
            path.append(current)
            current = previous[current]
        
        # Reverse the path to get it from start to end
        path.reverse()
        
        # If the path does not start with the start_id, no path was found
        if not path or path[0] != start_id:
            return [], float('inf')
        
        return path, distances[end_id]
    
    def find_reachable_points(self, start_id):
        if start_id not in self.navpoints:
            return []
        
        # Initialize visited set and queue for BFS
        visited = set()
        queue = [start_id]
        
        while queue:
            current_id = queue.pop(0)
            
            if current_id not in visited:
                visited.add(current_id)
                
                # Add all unvisited neighbors to the queue
                for segment in self.navsegments:
                    if segment.origin_number == current_id and segment.destination_number in self.navpoints:
                        if segment.destination_number not in visited:
                            queue.append(segment.destination_number)
        
        # Remove the start point from the result
        if start_id in visited:
            visited.remove(start_id)
        
        return list(visited)

if __name__ == "__main__":
    # Test the AirSpace class
    test_airspace = AirSpace("Catalunya")
    
    # Test loading from files (if files exist)
    try:
        success = test_airspace.load_from_files("Cat_nav.txt", "Cat_seg.txt", "Cat_aer.txt")
        if success:
            print(test_airspace)
            print(test_airspace.get_info())
            
            # Test getting neighbors of a node
            node_id = 5129  # GODOX from the example
            neighbors = test_airspace.get_neighbors(node_id)
            print(f"Neighbors of node {node_id}:")
            for neighbor in neighbors:
                print(f"  {neighbor}")
            
            # Test finding shortest path
            start_airport = "LEBL"  # Barcelona
            end_airport = "LEZG"    # Example destination
            
            # Get airport objects
            start = test_airspace.get_airport(start_airport)
            end = test_airspace.get_airport(end_airport)
            
            if start and end and start.sids and end.stars:
                # Use the first SID and STAR for the test
                start_id = start.sids[0]
                end_id = end.stars[0]
                
                path, cost = test_airspace.find_shortest_path(start_id, end_id)
                
                print(f"Shortest path from {start_airport} (SID: {start_id}) to {end_airport} (STAR: {end_id}):")
                print(f"  Cost: {cost}")
                print(f"  Path: {path}")
                
                # Print the names of the navigation points in the path
                path_names = []
                for node_id in path:
                    node = test_airspace.get_navpoint(node_id)
                    if node:
                        path_names.append(node.name)
                
                print(f"  Path names: {' -> '.join(path_names)}")
    except Exception as e:
        print(f"Test error: {e}")