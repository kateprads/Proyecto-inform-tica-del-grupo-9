# Part 1: Imports, Core Classes (Unchanged), and New Helper Classes
# ===== Add these to your imports at the top =====
from xml.etree.ElementTree import Element, SubElement, tostring, Comment
from xml.dom import minidom
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.patches as patches
import numpy as np
import math
import random
import time # For simulation time tracking
from datetime import datetime, timezone # For KML gx:Track timestamps

# Added for KML generation
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom # For pretty printing KML

# Optional: If your X, Y coordinates are NOT already Latitude/Longitude, you'll need pyproj.
# For example, if your input files use a projected coordinate system like UTM.
# If your 'latitude' and 'longitude' in nav.txt are already lat/lon, you don't need pyproj.
# If you need it, uncomment the line below and install it: pip install pyproj
# from pyproj import Transformer

# ===== Core Graph Classes (Unchanged from your original script) =====
class Node:
    """Node class for graph representation"""
    def __init__(self, name, x, y, number=None):
        self.name = name
        self.x = x
        self.y = y
        self.number = number
        self.neighbors = []
        self.is_airport = False
    
    def __str__(self):
        return f"Node({self.name}, {self.x}, {self.y})"

class Segment:
    """Segment class connecting two nodes"""
    def __init__(self, name, origin, destination, cost=None):
        self.name = name
        self.origin = origin
        self.destination = destination
        if cost is None:
            # Calculate Euclidean distance if cost not provided
            self.cost = math.sqrt((destination.x - origin.x)**2 + (destination.y - origin.y)**2)
        else:
            self.cost = cost
    
    def __str__(self):
        return f"Segment({self.name}, {self.origin.name} -> {self.destination.name}, {self.cost})"

class Graph:
    """Graph class to manage nodes and segments"""
    def __init__(self):
        self.nodes = {}  # Dictionary of nodes
        self.segments = {}  # Dictionary of segments
    
    def add_node(self, node):
        """Add a node to the graph"""
        self.nodes[node.name] = node
    
    def add_segment(self, segment):
        """Add a segment to the graph"""
        self.segments[segment.name] = segment
        # Add to neighbors list if not already there
        if segment.destination not in segment.origin.neighbors:
            segment.origin.neighbors.append(segment.destination)
    
    def find_node(self, name):
        """Find a node by name"""
        return self.nodes.get(name)
    
    def get_neighbors(self, node_name):
        """Get neighbors of a node"""
        node = self.nodes.get(node_name)
        return node.neighbors if node else []

class Path:
    """Path class to represent a path through the graph"""
    def __init__(self, nodes=None, cost=0):
        self.nodes = nodes if nodes else []
        self.cost = cost
    
    def add_node(self, node, segment_cost=0):
        """Add a node to the path"""
        self.nodes.append(node)
        self.cost += segment_cost

# ===== Navigation Classes (Unchanged from your original script) =====
class NavPoint:
    """Navigation Point class"""
    def __init__(self, number, name, latitude, longitude):
        self.number = number
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.is_airport = False
    
    def __str__(self):
        return f"NavPoint({self.number}, {self.name}, {self.latitude}, {self.longitude})"

class NavSegment:
    """Navigation Segment class"""
    def __init__(self, origin_number, destination_number, distance):
        self.origin_number = origin_number
        self.destination_number = destination_number
        self.distance = distance
    
    def __str__(self):
        return f"NavSegment({self.origin_number} -> {self.destination_number}, {self.distance}km)"

class NavAirport:
    """Navigation Airport class"""
    def __init__(self, name):
        self.name = name
        self.sids = []  # Standard Instrument Departures
        self.stars = []  # Standard Terminal Arrival Routes
    
    def add_sid(self, sid_id):
        if sid_id not in self.sids:
            self.sids.append(sid_id)
    
    def add_star(self, star_id):
        if star_id not in self.stars:
            self.stars.append(star_id)
    
    def __str__(self):
        return f"NavAirport({self.name}, SIDs: {len(self.sids)}, STARs: {len(self.stars)})"

class AirSpace:
    """Airspace class to manage navigation data"""
    def __init__(self, name):
        self.name = name
        self.navpoints = {}  # Dictionary of NavPoint objects
        self.navsegments = []  # List of NavSegment objects
        self.navairports = {}  # Dictionary of NavAirport objects
    
    def add_navpoint(self, navpoint):
        self.navpoints[navpoint.number] = navpoint
    
    def add_segment(self, segment):
        self.navsegments.append(segment)
    
    def add_airport(self, airport):
        self.navairports[airport.name] = airport
    
    def get_neighbors(self, node_id):
        """Get neighboring nodes"""
        neighbors = []
        for segment in self.navsegments:
            if segment.origin_number == node_id:
                neighbor = self.navpoints.get(segment.destination_number)
                if neighbor and neighbor not in neighbors:
                    neighbors.append(neighbor)
            elif segment.destination_number == node_id:
                neighbor = self.navpoints.get(segment.origin_number)
                if neighbor and neighbor not in neighbors:
                    neighbors.append(neighbor)
        return neighbors
    
    def get_segment(self, origin_id, dest_id):
        """Get segment between two nodes"""
        for segment in self.navsegments:
            if segment.origin_number == origin_id and segment.destination_number == dest_id:
                return segment
        return None
    
    def find_reachable_points(self, start_id):
        """Find all points reachable from start_id using BFS"""
        visited = set()
        queue = [start_id]
        reachable = set()
        
        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            
            visited.add(current_id)
            reachable.add(current_id)
            
            # Add neighbors to queue
            for segment in self.navsegments:
                if segment.origin_number == current_id:
                    if segment.destination_number not in visited:
                        queue.append(segment.destination_number)
        
        return reachable
    
    def find_shortest_path(self, start_id, end_id, excluded_points=None, excluded_segments=None):
        """Find shortest path using Dijkstra's algorithm with optional exclusions"""
        if excluded_points is None:
            excluded_points = set()
        if excluded_segments is None:
            excluded_segments = set() # Store as (origin_id, dest_id) tuples

        if start_id not in self.navpoints or end_id not in self.navpoints:
            return [], float('inf')
        
        if start_id in excluded_points or end_id in excluded_points:
            return [], float('inf')

        # Initialize distances and previous nodes
        distances = {node_id: float('inf') for node_id in self.navpoints}
        distances[start_id] = 0
        previous = {}
        unvisited = set(self.navpoints.keys())

        # Filter out excluded points from unvisited set
        unvisited = {node_id for node_id in unvisited if node_id not in excluded_points}
        
        while unvisited:
            # Find unvisited node with minimum distance
            if not unvisited: break # No more unvisited nodes
            current = min(unvisited, key=lambda x: distances[x])
            
            if distances[current] == float('inf'):
                break
            
            if current == end_id:
                break
            
            unvisited.remove(current)
            
            # Check neighbors
            for segment in self.navsegments:
                # Check segments for current node as origin
                if segment.origin_number == current:
                    neighbor = segment.destination_number
                    if (current, neighbor) in excluded_segments: # Check if this segment is excluded
                        continue
                    if neighbor in unvisited and neighbor not in excluded_points:
                        alt_distance = distances[current] + segment.distance
                        if alt_distance < distances[neighbor]:
                            distances[neighbor] = alt_distance
                            previous[neighbor] = current
                # Also check for segments where current node is destination (for bi-directional consideration)
                # Although Dijkstra generally expects uni-directional segments for simplicity,
                # if segments can be traversed both ways, this part is needed.
                # For this airspace model, segments are likely directed.
                # If bi-directional, add logic for segments where segment.destination_number == current.
                # For simplicity, assuming current segments are directed as (origin -> destination)
        
        # Reconstruct path
        if end_id not in previous and start_id != end_id:
            return [], float('inf')
        
        path = []
        current = end_id
        while current is not None:
            path.insert(0, current)
            current = previous.get(current)
        
        return path, distances[end_id]


# ===== Graph Algorithm Functions (Unchanged from your original script) =====
def AddNode(graph, node):
    """Add a node to the graph"""
    graph.add_node(node)

def AddSegment(graph, segment_name, origin_name, destination_name, cost=None):
    """Add a segment to the graph"""
    origin = graph.nodes.get(origin_name)
    destination = graph.nodes.get(destination_name)
    
    if not origin or not destination:
        raise ValueError(f"Origin '{origin_name}' or destination '{destination_name}' not found")
    
    if cost is None:
        cost = math.sqrt((destination.x - origin.x)**2 + (destination.y - origin.y)**2)
    
    segment = Segment(segment_name, origin, destination, cost)
    graph.add_segment(segment)

def GetClosest(graph, x, y):
    """Find the closest node to given coordinates"""
    min_distance = float('inf')
    closest_node = None
    
    for node in graph.nodes.values():
        distance = math.sqrt((node.x - x)**2 + (node.y - y)**2)
        if distance < min_distance:
            min_distance = distance
            closest_node = node
    
    return closest_node

def FindReachableNodes(graph, start_name):
    """Find all nodes reachable from start_name using BFS"""
    if start_name not in graph.nodes:
        return []
    
    visited = set()
    queue = [graph.nodes[start_name]]
    reachable = []
    
    while queue:
        current = queue.pop(0)
        if current.name in visited:
            continue
        
        visited.add(current.name)
        reachable.append(current)
        
        # Add neighbors to queue
        for neighbor in current.neighbors:
            if neighbor.name not in visited:
                queue.append(neighbor)
    
    return reachable[1:]  # Exclude the starting node

def FindShortestPath(graph, start_name, end_name):
    """Find shortest path using A* algorithm"""
    if start_name not in graph.nodes or end_name not in graph.nodes:
        return None
    
    start_node = graph.nodes[start_name]
    end_node = graph.nodes[end_name]
    
    def heuristic(node):
        return math.sqrt((node.x - end_node.x)**2 + (node.y - end_node.y)**2)
    
    open_set = [(0, start_node)]
    came_from = {}
    g_score = {node: float('inf') for node in graph.nodes.values()}
    g_score[start_node] = 0
    f_score = {node: float('inf') for node in graph.nodes.values()}
    f_score[start_node] = heuristic(start_node)
    
    while open_set:
        current = min(open_set, key=lambda x: f_score[x[1]])
        open_set.remove(current)
        current_node = current[1]
        
        if current_node == end_node:
            # Reconstruct path
            path_nodes = []
            total_cost = 0
            node = current_node
            
            while node in came_from:
                path_nodes.insert(0, node)
                prev_node = came_from[node]
                # Find segment cost
                for segment in graph.segments.values():
                    if segment.origin == prev_node and segment.destination == node:
                        total_cost += segment.cost
                        break
                node = prev_node
            path_nodes.insert(0, start_node)
            
            return Path(path_nodes, total_cost)
        
        for neighbor in current_node.neighbors:
            # Find segment cost
            segment_cost = 0
            for segment in graph.segments.values():
                if segment.origin == current_node and segment.destination == neighbor:
                    segment_cost = segment.cost
                    break
            
            tentative_g_score = g_score[current_node] + segment_cost
            
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current_node
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor)
                
                if (f_score[neighbor], neighbor) not in open_set:
                    open_set.append((f_score[neighbor], neighbor))
    
    return None

def Plot(graph, ax):
    """Plot the graph with arrows, distances, and node names - clean version"""
    # Plot segments with arrows
    for segment in graph.segments.values():
        # Draw arrow from origin to destination
        dx = segment.destination.x - segment.origin.x
        dy = segment.destination.y - segment.origin.y
        
        # Draw the arrow
        ax.annotate('', xy=(segment.destination.x, segment.destination.y), 
                   xytext=(segment.origin.x, segment.origin.y),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=1.5, alpha=0.7))
        
        # Add distance labels - smaller and cleaner
        mid_x = (segment.origin.x + segment.destination.x) / 2
        mid_y = (segment.origin.y + segment.destination.y) / 2
        ax.text(mid_x, mid_y, f"{segment.cost:.1f}", 
                color='red', fontsize=6, fontweight='normal', alpha=0.8)
    
    # Plot nodes with names - smaller and no background
    for node in graph.nodes.values():
        if hasattr(node, 'is_airport') and node.is_airport:
            ax.plot(node.x, node.y, 's', color='green', markersize=10)
            ax.text(node.x + 0.03, node.y + 0.03, node.name, 
                    color='green', fontsize=7, weight='bold', alpha=0.9)
        else:
            ax.plot(node.x, node.y, 'o', color='darkblue', markersize=8)
            ax.text(node.x + 0.03, node.y + 0.03, node.name, 
                    color='darkblue', fontsize=7, weight='normal', alpha=0.9)

def PlotNode(graph, node_name, ax):
    """Plot a specific node and its neighbors"""
    if node_name not in graph.nodes:
        return False
    
    node = graph.nodes[node_name]
    
    # Highlight the node
    ax.plot(node.x, node.y, 'ro', markersize=10)
    ax.text(node.x, node.y, f" {node.name}", 
            color='red', fontsize=10, weight='bold')
    
    # Highlight neighbors
    for neighbor in node.neighbors:
        ax.plot(neighbor.x, neighbor.y, 'yo', markersize=8)
        ax.text(neighbor.x, neighbor.y, f" {neighbor.name}", 
                color='orange', fontsize=9)
        
        # Highlight connecting segments
        ax.plot(
            [node.x, neighbor.x],
            [node.y, neighbor.y],
            'r-', linewidth=1.5
        )
    
    return True

def PlotPath(graph, path, ax):
    """Plot a path on the graph"""
    if not path or not path.nodes:
        return
    
    # Draw path segments
    for i in range(len(path.nodes) - 1):
        start = path.nodes[i]
        end = path.nodes[i + 1]
        ax.plot([start.x, end.x], [start.y, end.y], 'r-', linewidth=3)
    
    # Highlight path nodes
    for i, node in enumerate(path.nodes):
        if i == 0:  # Start node
            ax.plot(node.x, node.y, 'go', markersize=12)
            ax.text(node.x, node.y, f" {node.name}", 
                    color='green', fontsize=10, weight='bold')
        elif i == len(path.nodes) - 1:  # End node
            ax.plot(node.x, node.y, 'ro', markersize=12)
            ax.text(node.x, node.y, f" {node.name}", 
                    color='red', fontsize=10, weight='bold')
        else:  # Intermediate nodes
            ax.plot(node.x, node.y, 'mo', markersize=8)
            ax.text(node.x, node.y, f" {node.name}", 
                    color='magenta', fontsize=8)

def ReadGraphFromFile(file_path):
    """Read graph from file - supports multiple formats"""
    graph = Graph()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"Reading graph file: {file_path}")
        print(f"Total lines: {len(lines)}")
        
        mode = None
        nodes_added = 0
        segments_added = 0
        
        for line_num, line in enumerate(lines, 1):
            original_line = line
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            print(f"Line {line_num}: '{line}'")
            
            # Check for mode changes (original format)
            if line == "NODES":
                mode = "nodes"
                print("  -> Switched to NODES mode")
                continue
            elif line == "SEGMENTS":
                mode = "segments"
                print("  -> Switched to SEGMENTS mode")
                continue
            
            # Parse NODE lines (new format: NODE name x y)
            if line.upper().startswith("NODE "):
                parts = line.split()
                if len(parts) >= 4:
                    name = parts[1]
                    x = float(parts[2])
                    y = float(parts[3])
                    node = Node(name, x, y)
                    graph.add_node(node)
                    nodes_added += 1
                    print(f"  -> Added node: {name} at ({x}, {y})")
                else:
                    print(f"  -> Warning: Invalid NODE format on line {line_num}")
                continue
            
            # Parse SEGMENT lines (new format: SEGMENT name origin dest [cost])
            if line.upper().startswith("SEGMENT "):
                parts = line.split()
                if len(parts) >= 4:
                    segment_name = parts[1]
                    origin_name = parts[2]
                    dest_name = parts[3]
                    cost = float(parts[4]) if len(parts) > 4 else None
                    
                    try:
                        AddSegment(graph, segment_name, origin_name, dest_name, cost)
                        segments_added += 1
                        print(f"  -> Added segment: {segment_name} ({origin_name} -> {dest_name}, cost: {cost})")
                    except ValueError as e:
                        print(f"  -> Warning: {e}")
                else:
                    print(f"  -> Warning: Invalid SEGMENT format on line {line_num}")
                continue
            
            # Handle original format based on mode
            if mode == "nodes":
                parts = line.split()
                if len(parts) >= 3:
                    name = parts[0]
                    x = float(parts[1])
                    y = float(parts[2])
                    node = Node(name, x, y)
                    graph.add_node(node)
                    nodes_added += 1
                    print(f"  -> Added node (original format): {name} at ({x}, {y})")
            
            elif mode == "segments":
                parts = line.split()
                if len(parts) >= 3:
                    segment_name = parts[0]
                    origin_name = parts[1]
                    dest_name = parts[2]
                    cost = float(parts[3]) if len(parts) > 3 else None
                    
                    try:
                        AddSegment(graph, segment_name, origin_name, dest_name, cost)
                        segments_added += 1
                        print(f"  -> Added segment (original format): {segment_name} ({origin_name} -> {dest_name}, cost: {cost})")
                    except ValueError as e:
                        print(f"  -> Warning: {e}")
        
        print(f"Graph loading complete: {nodes_added} nodes, {segments_added} segments")
        
        if nodes_added == 0:
            print("Warning: No nodes were loaded!")
            print("File format should be either:")
            print("Format 1 (with NODE/SEGMENT keywords):")
            print("NODE A 10 10")
            print("NODE B 20 10")
            print("SEGMENT AB A B")
            print("\nFormat 2 (with NODES/SEGMENTS sections):")
            print("NODES")
            print("A 10 10")
            print("B 20 10")
            print("SEGMENTS")
            print("AB A B")
    
    except Exception as e:
        print(f"Error reading graph file: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return graph if nodes_added > 0 else None

# ===== File Parsing Functions (Unchanged from your original script) =====
def parse_navpoints_file(file_path):
    """Parse navigation points file"""
    navpoints = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split()
                    if len(parts) >= 4:
                        nav_id = int(parts[0])
                        name = parts[1]
                        latitude = float(parts[2])
                        longitude = float(parts[3])
                        
                        navpoint = NavPoint(nav_id, name, latitude, longitude)
                        navpoints[nav_id] = navpoint
                
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse navpoint line {line_num}: '{line}' - {e}")
                    continue
    
    except Exception as e:
        print(f"Error reading navpoints file: {e}")
    
    return navpoints

def parse_segments_file(file_path):
    """Parse segments file"""
    segments = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split()
                    if len(parts) >= 3:
                        origin_id = int(parts[0])
                        dest_id = int(parts[1])
                        distance = float(parts[2])
                        
                        segment = NavSegment(origin_id, dest_id, distance)
                        segments.append(segment)
                
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse segment line {line_num}: '{line}' - {e}")
                    continue
    
    except Exception as e:
        print(f"Error reading segments file: {e}")
    
    return segments

def parse_airports_file(file_path, navpoints):
    """Parse airports file"""
    airports = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_airport = None
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            # Check if it's an airport code (4 letters)
            if len(line) == 4 and line.isalpha() and line.isupper():
                current_airport = line
                airports[current_airport] = NavAirport(current_airport)
            
            elif current_airport:
                # Look for SIDs and STARs by matching with navpoint names
                if line.endswith('.D'):
                    # SID (departure)
                    sid_name = line[:-2]  # Remove .D
                    # Find corresponding navpoint
                    for nav_id, navpoint in navpoints.items():
                        if navpoint.name == sid_name:
                            airports[current_airport].add_sid(nav_id)
                            break
                
                elif line.endswith('.A'):
                    # STAR (arrival)
                    star_name = line[:-2]  # Remove .A
                    # Find corresponding navpoint
                    for nav_id, navpoint in navpoints.items():
                        if navpoint.name == star_name:
                            airports[current_airport].add_star(nav_id)
                            break
    
    except Exception as e:
        print(f"Error reading airports file: {e}")
    
    return airports

# ===== NEW HELPER CLASSES FOR ADVANCED FEATURES (Updated SimulatedAircraft for KML gx:Track) =====
class SimulatedAircraft:
    def __init__(self, aircraft_id, start_navpoint, end_navpoint, speed_kph=800):
        self.aircraft_id = aircraft_id
        self.start_navpoint = start_navpoint
        self.end_navpoint = end_navpoint
        self.current_navpoint = start_navpoint
        self.next_navpoint = None
        self.path_to_destination = []
        self.current_segment_progress = 0.0 # 0.0 to 1.0
        self.speed_kph = speed_kph
        self.is_active = True
        self.conflict_warning = False
        self.original_path = [] # For rerouting visualization
        self.rerouting = False
        self.altitude = random.randint(250, 400) * 100 # FL250 to FL400 (meters)
        self.icon_artist = None # Matplotlib artist for the aircraft icon
        self.trail_artist = None # Matplotlib artist for the trail
        self.text_artist = None # Matplotlib artist for aircraft ID text
        self.path_line_artist = None # Matplotlib artist for its full path

        self.current_trail_coords = [] # Stores recent coordinates for visual trail
        self.kml_track_data = [] # Stores (lon, lat, alt, timestamp) for KML gx:Track

    def calculate_path(self, airspace, destination_navpoint, excluded_points=None, excluded_segments=None):
        if excluded_points is None:
            excluded_points = set()
        if excluded_segments is None:
            excluded_segments = set()

        path_ids, cost = airspace.find_shortest_path(
            self.current_navpoint.number, destination_navpoint.number,
            excluded_points, excluded_segments
        )
        if path_ids and len(path_ids) > 1:
            self.path_to_destination = [airspace.navpoints[nid] for nid in path_ids]
            # If not already rerouting, store original path
            if not self.rerouting:
                self.original_path = list(self.path_to_destination)
            self.current_navpoint = self.path_to_destination[0]
            self.next_navpoint = self.path_to_destination[1]
            self.current_segment_progress = 0.0
            self.is_active = True
            return True
        self.is_active = False # No valid path found
        return False

    def update_position(self, delta_time_hours, current_sim_time_epoch):
        """
        Updates aircraft position and records data for KML gx:Track.
        current_sim_time_epoch: the epoch time of the current simulation step.
        """
        if not self.is_active or not self.path_to_destination or len(self.path_to_destination) < 2:
            return

        current_point = self.current_navpoint
        next_point = self.next_navpoint

        if not current_point or not next_point:
            self.is_active = False
            return

        # Calculate segment distance using Haversine formula for better accuracy with lat/lon
        R = 6371 # Radius of Earth in kilometers
        lat1_rad = math.radians(current_point.latitude)
        lon1_rad = math.radians(current_point.longitude)
        lat2_rad = math.radians(next_point.latitude)
        lon2_rad = math.radians(next_point.longitude)

        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad

        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        segment_distance_km = R * c

        if segment_distance_km == 0: # Avoid division by zero
            self.current_segment_progress = 1.0 # Immediately move to next point
        else:
            distance_covered_in_step = self.speed_kph * delta_time_hours
            progress_increase = distance_covered_in_step / segment_distance_km
            self.current_segment_progress += progress_increase

        current_lon, current_lat = self.get_current_coordinates()
        if current_lon is not None and current_lat is not None:
            self.current_trail_coords.append((current_lon, current_lat))
            # Keep trail short for performance/visuals
            if len(self.current_trail_coords) > 20: # Keep last 20 points
                self.current_trail_coords.pop(0)
            
            # Record data for KML gx:Track
            # KML requires ISO 8601 format with 'Z' for UTC or timezone offset
            dt_object = datetime.fromtimestamp(current_sim_time_epoch, tz=timezone.utc)
            timestamp_iso = dt_object.isoformat(timespec='seconds') + 'Z'
            self.kml_track_data.append((current_lon, current_lat, self.altitude, timestamp_iso))


        if self.current_segment_progress >= 1.0:
            # Move to the next point in the path
            try:
                current_index = self.path_to_destination.index(next_point)
                if current_index + 1 < len(self.path_to_destination):
                    self.current_navpoint = next_point
                    self.next_navpoint = self.path_to_destination[current_index + 1]
                    self.current_segment_progress = self.current_segment_progress - 1.0 # Carry over remaining progress
                    if self.current_segment_progress < 0: self.current_segment_progress = 0 # Ensure non-negative
                else:
                    # Reached final destination
                    self.current_navpoint = next_point
                    self.next_navpoint = None
                    self.path_to_destination = []
                    self.is_active = False
                    self.current_trail_coords = [] # Clear trail on arrival
                    self.rerouting = False # Reset rerouting status
            except ValueError:
                # This can happen if the path was rerouted and next_point is no longer in path_to_destination
                # This typically means the path was dynamically updated, so no need to deactivate
                # We should recalculate path if next_point is not found, but this is handled by dynamic_reroute_for_weather
                pass


    def get_current_coordinates(self):
        if not self.is_active or not self.current_navpoint:
            return None, None # Or sensible default if not active

        if not self.next_navpoint or self.current_segment_progress >= 1.0:
             # If at destination or about to move to next segment, return current_navpoint's coords
            return self.current_navpoint.longitude, self.current_navpoint.latitude

        p = self.current_segment_progress
        lon = self.current_navpoint.longitude + p * (self.next_navpoint.longitude - self.current_navpoint.longitude)
        lat = self.current_navpoint.latitude + p * (self.next_navpoint.latitude - self.current_navpoint.latitude)
        return lon, lat

    def get_future_coordinates(self, time_ahead_hours):
        # Predict future position based on current segment and assumed speed
        if not self.is_active or not self.path_to_destination or len(self.path_to_destination) < 2:
            return None, None

        current_point = self.current_navpoint
        next_point = self.next_navpoint

        # Calculate segment distance using Haversine formula
        R = 6371 # Radius of Earth in kilometers
        lat1_rad = math.radians(current_point.latitude)
        lon1_rad = math.radians(current_point.longitude)
        lat2_rad = math.radians(next_point.latitude)
        lon2_rad = math.radians(next_point.longitude)

        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad

        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        segment_distance_km = R * c

        if segment_distance_km == 0:
            return next_point.longitude, next_point.latitude # Already at next point

        distance_to_cover = self.speed_kph * time_ahead_hours
        future_progress = self.current_segment_progress + (distance_to_cover / segment_distance_km)

        if future_progress < 1.0:
            # Still on the current segment
            lon = current_point.longitude + future_progress * (next_point.longitude - current_point.longitude)
            lat = current_point.latitude + future_progress * (next_point.latitude - current_point.latitude)
            return lon, lat
        else:
            # Moved past the current segment, predict into next ones if path allows
            # This is simplified: a real simulation would iterate through multiple segments.
            # For this demo, we'll just predict to the next waypoint if it's within range
            current_index = self.path_to_destination.index(next_point)
            if current_index + 1 < len(self.path_to_destination):
                next_next_point = self.path_to_destination[current_index + 1]
                remaining_progress = future_progress - 1.0 # Progress into the next segment
                
                # Distance of next segment
                lat1_rad_nn = math.radians(next_point.latitude)
                lon1_rad_nn = math.radians(next_point.longitude)
                lat2_rad_nn = math.radians(next_next_point.latitude)
                lon2_rad_nn = math.radians(next_next_point.longitude)
                dlon_nn = lon2_rad_nn - lon1_rad_nn
                dlat_nn = lat2_rad_nn - lat1_rad_nn
                a_nn = math.sin(dlat_nn / 2)**2 + math.cos(lat1_rad_nn) * math.cos(lat2_rad_nn) * math.sin(dlon_nn / 2)**2
                c_nn = 2 * math.atan2(math.sqrt(a_nn), math.sqrt(1 - a_nn))
                next_segment_distance_km = R * c_nn

                if next_segment_distance_km == 0:
                    return next_next_point.longitude, next_next_point.latitude

                # Check if remaining_progress is too large (i.e., prediction goes beyond next segment)
                if remaining_progress * next_segment_distance_km > self.speed_kph * time_ahead_hours * 2: # Heuristic limit
                    return next_next_point.longitude, next_next_point.latitude # Limit prediction to next point
                
                lon = next_point.longitude + remaining_progress * (next_next_point.longitude - next_point.longitude)
                lat = next_point.latitude + remaining_progress * (next_next_point.latitude - next_next_point.latitude)
                return lon, lat
            else:
                return next_point.longitude, next_point.latitude # At final destination

# ===== NEW CLASS: 3D Airspace Zone =====
class AirspaceZone3D:
    """3D Airspace Zone with altitude bounds and controlled airspace types"""
    
    ZONE_TYPES = {
        'TMA': {'color': '#FF6B6B', 'alpha': 0.4, 'priority': 3},  # Terminal Maneuvering Area - Red
        'CTR': {'color': '#4ECDC4', 'alpha': 0.5, 'priority': 4},  # Control Zone - Teal
        'AWY': {'color': '#45B7D1', 'alpha': 0.3, 'priority': 1},  # Airways - Blue
        'RESTRICTED': {'color': '#FF4757', 'alpha': 0.7, 'priority': 5},  # Restricted - Dark Red
        'PROHIBITED': {'color': '#2F1B69', 'alpha': 0.8, 'priority': 6},  # Prohibited - Dark Purple
        'DANGER': {'color': '#FFA726', 'alpha': 0.6, 'priority': 5}  # Danger - Orange
    }
    
    def __init__(self, zone_id, name, zone_type, vertices_lon_lat, min_altitude_ft, max_altitude_ft, description=""):
        self.zone_id = zone_id
        self.name = name
        self.zone_type = zone_type.upper()
        self.vertices = list(vertices_lon_lat)  # List of (lon, lat) tuples
        self.min_altitude_ft = min_altitude_ft
        self.max_altitude_ft = max_altitude_ft
        self.description = description
        
        # Visual properties
        self.is_active = True
        self.flash_state = False
        self.last_flash_time = 0
        self.flash_duration = 0.5  # seconds
        self.aircraft_inside = set()  # Track which aircraft are inside
        
        # Matplotlib artists
        self.ground_patch = None
        self.ceiling_patch = None
        self.side_patches = []
        self.label_text = None
        
    def get_properties(self):
        """Get visual properties for this zone type"""
        return self.ZONE_TYPES.get(self.zone_type, self.ZONE_TYPES['AWY'])
    
    def contains_aircraft(self, aircraft):
        """Check if aircraft is inside this 3D zone"""
        lon, lat = aircraft.get_current_coordinates()
        if lon is None or lat is None:
            return False
            
        # Check altitude bounds
        aircraft_alt_ft = aircraft.altitude * 3.28084  # Convert meters to feet
        if not (self.min_altitude_ft <= aircraft_alt_ft <= self.max_altitude_ft):
            return False
            
        # Check horizontal bounds using point-in-polygon
        return self._point_in_polygon(lon, lat)
    
    def _point_in_polygon(self, lon, lat):
        """Ray casting algorithm for point-in-polygon test"""
        n = len(self.vertices)
        inside = False
        if n < 3:
            return False
            
        p1x, p1y = self.vertices[0]
        for i in range(n + 1):
            p2x, p2y = self.vertices[i % n]
            if min(p1y, p2y) < lat <= max(p1y, p2y):
                if lon <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (lat - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or lon <= xinters:
                        inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def trigger_flash_effect(self):
        """Trigger visual flash effect when aircraft enters zone"""
        self.flash_state = True
        self.last_flash_time = time.time()
    
    def update_flash_state(self):
        """Update flash animation state"""
        if self.flash_state and (time.time() - self.last_flash_time) > self.flash_duration:
            self.flash_state = not self.flash_state
            self.last_flash_time = time.time()
    
    def get_kml_polygon_3d(self):
        """Generate KML 3D polygon representation"""
        # Ensure polygon is closed
        coords_ground = self.vertices + [self.vertices[0]] if self.vertices else []
        coords_ceiling = [(lon, lat, self.max_altitude_ft * 0.3048) for lon, lat in coords_ground]  # Convert ft to meters
        coords_ground_with_alt = [(lon, lat, self.min_altitude_ft * 0.3048) for lon, lat in coords_ground]
        
        props = self.get_properties()
        kml_color = self._rgb_to_kml_color(props['color'], props['alpha'])
        
        return f"""
    <Placemark>
        <name>{self.name} ({self.zone_type})</name>
        <description><![CDATA[
            Zone Type: {self.zone_type}<br/>
            Altitude Range: {self.min_altitude_ft:,} - {self.max_altitude_ft:,} ft<br/>
            Description: {self.description}
        ]]></description>
        <styleUrl>#{self.zone_type.lower()}_zone_style</styleUrl>
        <Polygon>
            <extrude>1</extrude>
            <altitudeMode>absolute</altitudeMode>
            <outerBoundaryIs>
                <LinearRing>
                    <coordinates>
                        {' '.join([f"{lon},{lat},{alt}" for lon, lat, alt in coords_ceiling])}
                    </coordinates>
                </LinearRing>
            </outerBoundaryIs>
        </Polygon>
    </Placemark>"""
    
    def _rgb_to_kml_color(self, hex_color, alpha):
        """Convert hex color to KML AABBGGRR format"""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        a = int(alpha * 255)
        return f"{a:02x}{b:02x}{g:02x}{r:02x}"

# ===== NEW CLASS: Enhanced Aircraft with Custom Models =====
class EnhancedAircraft(SimulatedAircraft):
    """Enhanced aircraft with custom icons, 3D models, and advanced behaviors"""
    
    AIRCRAFT_TYPES = {
        'AIRLINER': {
            'icon': 'http://maps.google.com/mapfiles/kml/shapes/airports.png',
            'speed_range': (450, 550),  # kph
            'cruise_alt_range': (30000, 42000),  # feet
            'color': '#2196F3',
            'size_scale': 1.2
        },
        'BUSINESS_JET': {
            'icon': 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png',
            'speed_range': (400, 500),
            'cruise_alt_range': (35000, 45000),
            'color': '#9C27B0',
            'size_scale': 0.8
        },
        'FIGHTER_JET': {
            'icon': 'http://maps.google.com/mapfiles/kml/shapes/target.png',
            'speed_range': (800, 1200),
            'cruise_alt_range': (25000, 50000),
            'color': '#F44336',
            'size_scale': 1.0
        },
        'HELICOPTER': {
            'icon': 'http://maps.google.com/mapfiles/kml/shapes/helicopter.png',
            'speed_range': (150, 250),
            'cruise_alt_range': (1000, 8000),
            'color': '#4CAF50',
            'size_scale': 0.9
        }
    }
    
    def __init__(self, aircraft_id, start_navpoint, end_navpoint, aircraft_type='AIRLINER'):
        self.aircraft_type = aircraft_type
        type_info = self.AIRCRAFT_TYPES.get(aircraft_type, self.AIRCRAFT_TYPES['AIRLINER'])
        
        # Set speed based on aircraft type
        speed_kph = random.randint(*type_info['speed_range'])
        super().__init__(aircraft_id, start_navpoint, end_navpoint, speed_kph)
        
        # Set altitude based on aircraft type
        alt_range = type_info['cruise_alt_range']
        self.altitude = random.randint(*alt_range) * 0.3048  # Convert feet to meters
        
        # Visual properties
        self.base_scale = type_info['size_scale']
        self.current_scale = self.base_scale
        self.color = type_info['color']
        self.in_restricted_zone = False
        self.firejet_warning_active = False
        self.emergency_reroute_attempts = 0
        
        # Enhanced trail for different aircraft types
        self.max_trail_length = 30 if aircraft_type == 'FIGHTER_JET' else 20
        
    def get_aircraft_info(self):
        """Get aircraft type information"""
        return self.AIRCRAFT_TYPES.get(self.aircraft_type, self.AIRCRAFT_TYPES['AIRLINER'])
    
    def update_scale_by_altitude(self):
        """Dynamically scale aircraft icon based on altitude"""
        # Scale factor: higher altitude = smaller icon (simulating distance)
        alt_ft = self.altitude * 3.28084
        scale_factor = max(0.3, min(1.5, 1.0 - (alt_ft - 20000) / 50000))
        self.current_scale = self.base_scale * scale_factor
    
    def activate_firejet_warning(self):
        """Activate firejet warning for restricted zone violation"""
        self.firejet_warning_active = True
        self.in_restricted_zone = True
        print(f"🚨 FIREJET WARNING: {self.aircraft_id} entering restricted airspace!")
    
    def deactivate_firejet_warning(self):
        """Deactivate firejet warning"""
        self.firejet_warning_active = False
        self.in_restricted_zone = False
    
    def get_kml_model_3d(self):
        """Generate KML 3D model representation"""
        lon, lat = self.get_current_coordinates()
        if lon is None or lat is None:
            return ""
            
        type_info = self.get_aircraft_info()
        heading = self._calculate_heading()
        
        return f"""
    <Placemark>
        <name>{self.aircraft_id}</name>
        <description><![CDATA[
            Aircraft Type: {self.aircraft_type}<br/>
            Speed: {self.speed_kph} kph<br/>
            Altitude: FL{self.altitude * 3.28084 / 100:.0f}<br/>
            Status: {'⚠️ RESTRICTED ZONE' if self.in_restricted_zone else '✅ Normal Flight'}
        ]]></description>
        <Point>
            <coordinates>{lon},{lat},{self.altitude}</coordinates>
        </Point>
        <styleUrl>#{self.aircraft_type.lower()}_aircraft_style</styleUrl>
    </Placemark>"""
    
    def _calculate_heading(self):
        """Calculate aircraft heading based on direction to next waypoint"""
        if not self.next_navpoint:
            return 0
            
        current_lon, current_lat = self.get_current_coordinates()
        next_lon, next_lat = self.next_navpoint.longitude, self.next_navpoint.latitude
        
        # Calculate bearing
        dlon = math.radians(next_lon - current_lon)
        lat1 = math.radians(current_lat)
        lat2 = math.radians(next_lat)
        
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        
        bearing = math.degrees(math.atan2(y, x))
        return (bearing + 360) % 360  # Normalize to 0-360

class WeatherZone:
    def __init__(self, name, vertices_lon_lat, severity="light", speed_kph=20, direction_degrees=45):
        self.name = name
        self.vertices = list(vertices_lon_lat) # List of (lon, lat) tuples
        self.severity = severity # "light", "moderate", "severe"
        self.speed_kph = speed_kph
        self.direction_rad = math.radians(direction_degrees)
        self.plot_artist = None # Matplotlib artist for the polygon

    def update_position(self, delta_time_hours):
        distance_moved_km = self.speed_kph * delta_time_hours
        
        # Approximating degrees per km for movement.
        # This is a simplification; for real applications,
        # movement in lat/lon is more complex (e.g., using Vincenty's formula or a library like geopy).
        # Assuming ~111 km per degree for simplicity.
        degrees_per_km_approx = 1 / 111.0 

        delta_lon_degrees = distance_moved_km * degrees_per_km_approx * math.sin(self.direction_rad)
        delta_lat_degrees = distance_moved_km * degrees_per_km_approx * math.cos(self.direction_rad)
        
        new_vertices = []
        for lon, lat in self.vertices:
            new_vertices.append((lon + delta_lon_degrees, lat + delta_lat_degrees))
        self.vertices = new_vertices

    def get_mpl_polygon_coords(self):
        # Matplotlib Polygon expects (x, y) which corresponds to (longitude, latitude)
        return np.array(self.vertices)

    def contains_point(self, lon, lat):
        # Ray casting algorithm for point-in-polygon
        # Reference: https://en.wikipedia.org/wiki/Point_in_polygon
        n = len(self.vertices)
        inside = False
        if n < 3: return False # Need at least 3 vertices for a polygon

        # Loop through each edge of the polygon
        p1x, p1y = self.vertices[0]
        for i in range(n + 1):
            p2x, p2y = self.vertices[i % n] # Connect last vertex to first for closing loop

            # Check if the ray from (lon,lat) intersects the edge (p1, p2)
            # This simplified logic assumes the ray goes horizontally to the right
            if (p1y <= lat < p2y) or (p2y <= lat < p1y):
                if lon < (p2x - p1x) * (lat - p1y) / (p2y - p1y) + p1x:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside
# Part 2: KML Generation Functions

# ===== KML Generation Functions (Added as per your request) =====
# Part 2: KML Generation Functions (Updated for gx:Track)

# ===== KML Generation Functions (Added as per your request) =====
def _generate_kml_header(doc_name="Google Earth Display"):
    # Styles for KML elements (you can customize these styles and icons)
    # Note: gx (Google Extension) namespace is needed for gx:Track
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"
  xmlns:gx="http://www.google.com/kml/ext/2.2">
  <Document>
    <name>{doc_name}</name>
    <Style id="airportIcon">
      <IconStyle>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/pal2/icon18.png</href> </Icon>
        <scale>1.2</scale>
      </IconStyle>
      <LabelStyle>
        <scale>1.2</scale>
      </LabelStyle>
    </Style>
    <Style id="navpointIcon">
      <IconStyle>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/pal2/icon21.png</href> </Icon>
        <scale>0.8</scale>
      </IconStyle>
      <LabelStyle>
        <scale>0.8</scale>
      </LabelStyle>
    </Style>
    <Style id="pathLine">
      <LineStyle>
        <color>ff0000ff</color> <width>4</width>
      </LineStyle>
    </Style>
    <Style id="airwayLine">
      <LineStyle>
        <color>ffbf005f</color> <width>2</width>
      </LineStyle>
    </Style>
    <Style id="conflictArea">
        <LineStyle>
            <color>ff0000ff</color> <width>3</width>
        </LineStyle>
        <PolyStyle>
            <color>4c0000ff</color> <fill>1</fill>
            <outline>1</outline>
        </PolyStyle>
    </Style>
    <Style id="weatherZoneLight">
        <LineStyle>
            <color>ff00ffff</color> <width>2</width>
        </LineStyle>
        <PolyStyle>
            <color>4c00ffff</color> <fill>1</fill>
            <outline>1</outline>
        </PolyStyle>
    </Style>
    <Style id="weatherZoneModerate">
        <LineStyle>
            <color>ff00aaff</color> <width>3</width>
        </LineStyle>
        <PolyStyle>
            <color>7f00aaff</color> <fill>1</fill>
            <outline>1</outline>
        </PolyStyle>
    </Style>
    <Style id="weatherZoneSevere">
        <LineStyle>
            <color>ff0000ff</color> <width>4</width>
        </LineStyle>
        <PolyStyle>
            <color>990000ff</color> <fill>1</fill>
            <outline>1</outline>
        </PolyStyle>
    </Style>
    <Style id="aircraftTrack">
      <IconStyle>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href> </Icon>
        <scale>0.7</scale>
      </IconStyle>
      <LabelStyle>
        <scale>0</scale> </LabelStyle>
      <LineStyle>
        <color>ff00aaff</color> <width>2</width>
      </LineStyle>
    </Style>
    <Style id="tma_zone_style">
        <LineStyle>
            <color>4c6bffff</color> <width>3</width>
        </LineStyle>
        <PolyStyle>
            <color>4c6bffff</color> <fill>1</fill>
            <outline>1</outline>
        </PolyStyle>
    </Style>
    <Style id="restricted_zone_style">
        <LineStyle>
            <color>b34757ff</color> <width>4</width>
        </LineStyle>
        <PolyStyle>
            <color>b34757ff</color> <fill>1</fill>
            <outline>1</outline>
        </PolyStyle>
    </Style>
    <Style id="airliner_aircraft_style">
      <IconStyle>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/shapes/airports.png</href>
        </Icon>
        <scale>1.2</scale>
      </IconStyle>
    </Style>
    <Style id="fighter_jet_aircraft_style">
      <IconStyle>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/shapes/target.png</href>
        </Icon>
        <scale>1.0</scale>
      </IconStyle>
    </Style>
    """

def _generate_kml_footer():
    return """  </Document>
</kml>"""

def _generate_kml_placemark_point(name, description, longitude, latitude, altitude=0, style_url=""):
    style_tag = f"<styleUrl>#{style_url}</styleUrl>" if style_url else ""
    return f"""    <Placemark>
      <name>{name}</name>
      <description>{description}</description>
      {style_tag}
      <Point>
        <coordinates>{longitude},{latitude},{altitude}</coordinates>
      </Point>
    </Placemark>"""

def _generate_kml_linestring(name, description, coordinates_list, altitude=0, style_url="", altitude_mode="clampToGround"):
    # coordinates_list is a list of (longitude, latitude) tuples
    # KML coordinates are Longitude, Latitude, Altitude
    coords_str = " ".join([f"{lon},{lat},{altitude}" for lon, lat in coordinates_list])
    style_tag = f"<styleUrl>#{style_url}</styleUrl>" if style_url else ""
    return f"""    <Placemark>
      <name>{name}</name>
      <description>{description}</description>
      {style_tag}
      <LineString>
        <extrude>1</extrude>
        <tessellate>1</tessellate>
        <altitudeMode>{altitude_mode}</altitudeMode>
        <coordinates>{coords_str}</coordinates>
      </LineString>
    </Placemark>"""

def _generate_kml_polygon(name, description, coordinates_list, altitude=0, style_url="", altitude_mode="clampToGround"):
    # coordinates_list is a list of (longitude, latitude) tuples forming the boundary
    # The first and last coordinate should be the same to close the polygon
    coords_str = " ".join([f"{lon},{lat},{altitude}" for lon, lat in coordinates_list])
    style_tag = f"<styleUrl>#{style_url}</styleUrl>" if style_url else ""
    return f"""    <Placemark>
        <name>{name}</name>
        <description>{description}</description>
        {style_tag}
        <Polygon>
            <extrude>1</extrude>
            <altitudeMode>{altitude_mode}</altitudeMode>
            <outerBoundaryIs>
                <LinearRing>
                    <coordinates>{coords_str}</coordinates>
                </LinearRing>
            </outerBoundaryIs>
        </Polygon>
    </Placemark>"""

def _generate_kml_gxtrack(name, description, track_data, style_url="aircraftTrack"):
    # track_data is a list of (lon, lat, alt, timestamp_iso) tuples
    when_tags = "\n".join([f"<when>{t}</when>" for lon, lat, alt, t in track_data])
    coord_tags = "\n".join([f"<gx:coord>{lon} {lat} {alt}</gx:coord>" for lon, lat, alt, t in track_data])
    
    # Use the gx namespace for track elements
    return f"""    <Placemark>
      <name>{name}</name>
      <description>{description}</description>
      <styleUrl>#{style_url}</styleUrl>
      <gx:Track>
        {when_tags}
        {coord_tags}
      </gx:Track>
    </Placemark>"""
# Part 3: CompleteAirspaceInterface Class (Initialization and UI Setup)

# ===== Main Interface Class =====
class CompleteAirspaceInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Complete Airspace Navigation Tool")
        self.root.geometry("1500x1000")  # Increased size for bigger graphs
        
        # Data storage
        self.airspaces = {
            "Catalunya": None,
            "España": None,
            "Europe": None
        }
        self.current_airspace = None
        self.current_airspace_name = tk.StringVar(value="None")
        self.graph = None  # For custom graphs
        
        # UI variables
        self.source_node = tk.StringVar()
        self.dest_node = tk.StringVar()
        self.airport = tk.StringVar()
        self.search_node = tk.StringVar()

        # New state variables for advanced features
        self.simulated_aircrafts = []
        self.weather_zones = []
        self.simulation_active = False
        self.simulation_speed = tk.DoubleVar(value=1.0) # For animation speed control
        self.animation_job = None # To store after_id for animation loop
        self.animation_interval_ms = 50  # 50ms for smooth animation
        
        # Matplotlib artists for dynamic plotting (to allow efficient updates)
        # Store dicts of artists for each aircraft/weather zone
        self.aircraft_plot_artists = {} # {aircraft_id: {'icon': artist, 'trail': artist, 'text': artist, 'path_line': artist}}
        self.conflict_plot_artists = [] # List of artists for conflict bubbles/lines
        self.weather_plot_artists = {} # {weather_zone_name: artist}

        # Last plot update time for delta_time calculation
        self.last_update_time = time.time() 

        # Current path being displayed for KML export (from show_shortest_path)
        self.current_display_path_nodes = []
        self.current_display_path_cost = float('inf')
        
        # 3D Airspace zones (NEW)
        self.airspace_zones_3d = []
        
        # Configure cool color theme
        self.setup_cool_theme()
        
        self.setup_ui()
    
    def setup_cool_theme(self):
        """Setup a cool blue/cyan color theme"""
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Cool color palette
        self.bg_primary = "#1e3a5f"      # Dark blue
        self.bg_secondary = "#2c5282"    # Medium blue
        self.bg_light = "#bee3f8"        # Light blue
        self.bg_very_light = "#e6f3ff"   # Very light blue
        self.accent_cyan = "#0bc5ea"     # Bright cyan
        self.accent_teal = "#319795"     # Teal
        self.text_dark = "#1a202c"       # Dark text
        self.text_light = "#ffffff"      # White text
        
        # Configure root window
        self.root.configure(background=self.bg_light)
        
        # Configure ttk styles with cool colors
        self.style.configure("TFrame", 
                            background=self.bg_light, 
                            relief="flat")
        
        self.style.configure("TLabel", 
                            background=self.bg_light, 
                            foreground=self.text_dark,
                            font=("Segoe UI", 10))
        
        self.style.configure("TButton", 
                            background=self.accent_cyan,
                            foreground=self.text_light,
                            font=("Segoe UI", 9, "bold"),
                            padding=(10, 5),
                            relief="flat")
        
        self.style.map("TButton",
                      background=[('active', self.accent_teal),
                                ('pressed', self.bg_primary)])
        
        self.style.configure("TLabelFrame", 
                            background=self.bg_light,
                            foreground=self.text_dark,
                            font=("Segoe UI", 10, "bold"))
        
        self.style.configure("TLabelFrame.Label",
                            background=self.bg_light,
                            foreground=self.bg_primary,
                            font=("Segoe UI", 11, "bold"))
        
        self.style.configure("TNotebook", 
                            background=self.bg_light,
                            tabposition='n')
        
        self.style.configure("TNotebook.Tab", 
                            background=self.bg_secondary,
                            foreground=self.text_light,
                            padding=(15, 8),
                            font=("Segoe UI", 10, "bold"))
        
        self.style.map("TNotebook.Tab",
                      background=[('selected', self.accent_cyan),
                                ('active', self.accent_teal)])
        
        self.style.configure("TCombobox",
                            fieldbackground=self.bg_very_light,
                            background=self.bg_secondary,
                            foreground=self.text_dark,
                            font=("Segoe UI", 9))
        
        self.style.configure("TEntry",
                            fieldbackground=self.bg_very_light,
                            foreground=self.text_dark,
                            font=("Segoe UI", 9))
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook with tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Airspace Analysis
        self.airspace_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.airspace_tab, text="Airspace Analysis")
        self.setup_airspace_tab()
        
        # Tab 2: Graph Editor
        self.graph_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.graph_tab, text="Graph Editor")
        self.setup_graph_tab()

        # Tab 3: Advanced Features (New Tab)
        self.advanced_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.advanced_tab, text="Advanced Features")
        self.setup_advanced_tab()

        # Bind tab change event to refresh relevant visualization
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)
    
    def setup_airspace_tab(self):
        """Setup the airspace analysis tab"""
        # Control panel
        control_frame = ttk.Frame(self.airspace_tab)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Airspace selection
        airspace_frame = ttk.LabelFrame(control_frame, text="Airspace Selection")
        airspace_frame.pack(fill=tk.X, padx=5, pady=5)
        
        airspace_buttons_frame = ttk.Frame(airspace_frame)
        airspace_buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(airspace_buttons_frame, text="Load Catalunya", 
                  command=lambda: self.load_airspace("Catalunya")).pack(side=tk.LEFT, padx=5)
        ttk.Button(airspace_buttons_frame, text="Load España", 
                  command=lambda: self.load_airspace("España")).pack(side=tk.LEFT, padx=5)
        ttk.Button(airspace_buttons_frame, text="Load Europe", 
                  command=lambda: self.load_airspace("Europe")).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(airspace_buttons_frame, text="Current:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Label(airspace_buttons_frame, textvariable=self.current_airspace_name).pack(side=tk.LEFT)
        
        # Analysis controls
        analysis_frame = ttk.LabelFrame(control_frame, text="Analysis Functions")
        analysis_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Node selection
        node_selection_frame = ttk.Frame(analysis_frame)
        node_selection_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(node_selection_frame, text="Source:").pack(side=tk.LEFT, padx=5)
        self.source_combo = ttk.Combobox(node_selection_frame, textvariable=self.source_node, width=10)
        self.source_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(node_selection_frame, text="Destination:").pack(side=tk.LEFT, padx=5)
        self.dest_combo = ttk.Combobox(node_selection_frame, textvariable=self.dest_node, width=10)
        self.dest_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(node_selection_frame, text="Airport:").pack(side=tk.LEFT, padx=5)
        self.airport_combo = ttk.Combobox(node_selection_frame, textvariable=self.airport, width=10)
        self.airport_combo.pack(side=tk.LEFT, padx=5)
        
        # Function buttons
        function_buttons_frame = ttk.Frame(analysis_frame)
        function_buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(function_buttons_frame, text="Show Complete Airspace", 
                  command=self.show_complete_airspace).pack(side=tk.LEFT, padx=5)
        ttk.Button(function_buttons_frame, text="Show Neighbors", 
                  command=self.show_neighbors).pack(side=tk.LEFT, padx=5)
        ttk.Button(function_buttons_frame, text="Show Reachability", 
                  command=self.show_reachability).pack(side=tk.LEFT, padx=5)
        ttk.Button(function_buttons_frame, text="Show Shortest Path", 
                  command=self.show_shortest_path).pack(side=tk.LEFT, padx=5)
        ttk.Button(function_buttons_frame, text="Show Airport Info", 
                  command=self.show_airport_info).pack(side=tk.LEFT, padx=5)

        # KML Export buttons (Integrated into Airspace Analysis Tab)
        kml_export_frame = ttk.LabelFrame(control_frame, text="Google Earth Export")
        kml_export_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(kml_export_frame, text="Export All Airspace to KML", 
                   command=self.generate_kml_airspace).pack(side=tk.LEFT, padx=5)
        ttk.Button(kml_export_frame, text="Export Shortest Path to KML", 
                   command=self.generate_kml_path).pack(side=tk.LEFT, padx=5)
        
        # Visualization area
        self.setup_visualization_area(self.airspace_tab, "airspace")
    
    def setup_graph_tab(self):
        """Setup the graph editor tab"""
        # Control panel
        control_frame = ttk.Frame(self.graph_tab)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Graph management
        graph_frame = ttk.LabelFrame(control_frame, text="Graph Management")
        graph_frame.pack(fill=tk.X, padx=5, pady=5)
        
        graph_buttons_frame = ttk.Frame(graph_frame)
        graph_buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(graph_buttons_frame, text="Example Graph", 
                  command=self.load_example_graph).pack(side=tk.LEFT, padx=5)
        ttk.Button(graph_buttons_frame, text="Custom Graph", 
                  command=self.load_custom_graph).pack(side=tk.LEFT, padx=5)
        ttk.Button(graph_buttons_frame, text="Load from File", 
                  command=self.load_graph_from_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(graph_buttons_frame, text="Save Graph", 
                  command=self.save_graph).pack(side=tk.LEFT, padx=5)
        
        # Search and edit
        edit_frame = ttk.LabelFrame(control_frame, text="Search and Edit")
        edit_frame.pack(fill=tk.X, padx=5, pady=5)
        
        search_frame = ttk.Frame(edit_frame)
        search_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(search_frame, text="Search Node:").pack(side=tk.LEFT, padx=5)
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_node, width=15)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(search_frame, text="Show Node", 
                  command=self.show_graph_node).pack(side=tk.LEFT, padx=5)
        
        edit_buttons_frame = ttk.Frame(edit_frame)
        edit_buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(edit_buttons_frame, text="Add Node", 
                  command=self.add_node).pack(side=tk.LEFT, padx=5)
        ttk.Button(edit_buttons_frame, text="Add Segment", 
                  command=self.add_segment).pack(side=tk.LEFT, padx=5)
        ttk.Button(edit_buttons_frame, text="Delete Node", 
                  command=self.delete_node).pack(side=tk.LEFT, padx=5)
        ttk.Button(edit_buttons_frame, text="Find Closest", 
                  command=self.find_closest_node).pack(side=tk.LEFT, padx=5)
        ttk.Button(edit_buttons_frame, text="Show Reachable", 
                  command=self.show_graph_reachable).pack(side=tk.LEFT, padx=5)
        ttk.Button(edit_buttons_frame, text="Shortest Path", 
                  command=self.show_graph_shortest_path).pack(side=tk.LEFT, padx=5)
        
        # Visualization area
        self.setup_visualization_area(self.graph_tab, "graph")

    def setup_advanced_tab(self):
        """Setup the advanced features tab with animation controls and enhanced features"""
        control_frame = ttk.Frame(self.advanced_tab)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Dynamic Air Traffic Flow Simulation & Conflict Prediction
        traffic_frame = ttk.LabelFrame(control_frame, text="Dynamic Air Traffic Simulation & Conflicts")
        traffic_frame.pack(fill=tk.X, padx=5, pady=5)

        traffic_buttons_frame = ttk.Frame(traffic_frame)
        traffic_buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        # Enhanced Aircraft section (NEW)
        enhanced_aircraft_frame = ttk.LabelFrame(control_frame, text="✈️ Enhanced Aircraft Models")
        enhanced_aircraft_frame.pack(fill=tk.X, padx=5, pady=5)

        
        
        ttk.Button(traffic_buttons_frame, text="Spawn Aircraft", 
                   command=self.spawn_aircraft).pack(side=tk.LEFT, padx=5)
        ttk.Button(traffic_buttons_frame, text="Start Simulation", 
                   command=self.start_traffic_simulation).pack(side=tk.LEFT, padx=5)
        ttk.Button(traffic_buttons_frame, text="Stop Simulation", 
                   command=self.stop_traffic_simulation).pack(side=tk.LEFT, padx=5)
        ttk.Button(traffic_buttons_frame, text="Clear Aircraft", 
                   command=self.clear_aircraft_simulation).pack(side=tk.LEFT, padx=5)

        speed_frame = ttk.Frame(traffic_frame)
        speed_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(speed_frame, text="Sim Speed:").pack(side=tk.LEFT, padx=(5, 2))
        ttk.Scale(speed_frame, from_=0.1, to=10.0, orient=tk.HORIZONTAL,
                  variable=self.simulation_speed, command=self.update_simulation_speed).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(speed_frame, textvariable=self.simulation_speed, width=5).pack(side=tk.LEFT)

        enhanced_buttons_frame = ttk.Frame(enhanced_aircraft_frame)
        enhanced_buttons_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(enhanced_buttons_frame, text="Spawn Enhanced Aircraft", 
           command=self.spawn_enhanced_aircraft).pack(side=tk.LEFT, padx=5)
        ttk.Button(enhanced_buttons_frame, text="Export 3D Models KML", 
           command=self.generate_kml_enhanced_aircraft_models).pack(side=tk.LEFT, padx=5)
        # Interception Scenario section (NEW)
        interception_frame = ttk.LabelFrame(control_frame, text="🎯 Interception Scenarios")
        interception_frame.pack(fill=tk.X, padx=5, pady=5)

        interception_buttons_frame = ttk.Frame(interception_frame)
        interception_buttons_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(interception_buttons_frame, text="Run Interception Scenario", 
           command=self.simulate_interception_scenario).pack(side=tk.LEFT, padx=5)
        ttk.Button(interception_buttons_frame, text="Export Scenario KML", 
           command=self.export_interception_scenario_kml).pack(side=tk.LEFT, padx=5)

        # Procedural Weather Overlay & Dynamic Rerouting
        weather_frame = ttk.LabelFrame(control_frame, text="Procedural Weather Overlay & Rerouting")
        weather_frame.pack(fill=tk.X, padx=5, pady=5)

        weather_buttons_frame = ttk.Frame(weather_frame)
        weather_buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(weather_buttons_frame, text="Generate Weather", 
                   command=self.generate_weather_zones).pack(side=tk.LEFT, padx=5)
        ttk.Button(weather_buttons_frame, text="Clear Weather", 
                   command=self.clear_weather_zones).pack(side=tk.LEFT, padx=5)
        
        # 3D Airspace Zones section (NEW)
        zones_3d_frame = ttk.LabelFrame(control_frame, text="🏢 3D Airspace Zones & Restricted Areas")
        zones_3d_frame.pack(fill=tk.X, padx=5, pady=5)

        zones_buttons_frame = ttk.Frame(zones_3d_frame)
        zones_buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(zones_buttons_frame, text="Setup 3D Zones", 
                   command=self.setup_3d_airspace_zones).pack(side=tk.LEFT, padx=5)
        ttk.Button(zones_buttons_frame, text="Export 3D Zones KML", 
                   command=self.generate_kml_3d_airspace_zones).pack(side=tk.LEFT, padx=5)
        
        # Enhanced Aircraft section (NEW)
        enhanced_aircraft_frame = ttk.LabelFrame(control_frame, text="✈️ Enhanced Aircraft Models")
        enhanced_aircraft_frame.pack(fill=tk.X, padx=5, pady=5)

        enhanced_buttons_frame = ttk.Frame(enhanced_aircraft_frame)
        enhanced_buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(enhanced_buttons_frame, text="Spawn Enhanced Aircraft", 
                   command=self.spawn_enhanced_aircraft).pack(side=tk.LEFT, padx=5)
        ttk.Button(enhanced_buttons_frame, text="Export 3D Models KML", 
                   command=self.generate_kml_enhanced_aircraft_models).pack(side=tk.LEFT, padx=5)
        
        # KML Export for Simulation
        sim_kml_export_frame = ttk.LabelFrame(control_frame, text="Simulation Google Earth Export")
        sim_kml_export_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(sim_kml_export_frame, text="Export Aircraft Tracks to KML", 
                   command=self.generate_kml_sim_aircraft_tracks).pack(side=tk.LEFT, padx=5)
        ttk.Button(sim_kml_export_frame, text="Export Current Weather to KML", 
                   command=self.generate_kml_sim_weather).pack(side=tk.LEFT, padx=5)

        # Visualization area for advanced features (reusing airspace visualization)
        self.setup_visualization_area(self.advanced_tab, "advanced")
    def generate_kml_sim_aircraft_tracks(self):
        """Generates a KML file with the simulated aircraft tracks (gx:Track)."""
        if not self.simulated_aircrafts:
            messagebox.showwarning("No Aircraft", "No simulated aircraft tracks to export. Run the simulation first.")
            return

        kml_content = []
        kml_content.append(_generate_kml_header("Simulated Aircraft Tracks"))

        tracks_folder = SubElement(Element("Folder"), "Folder")
        SubElement(tracks_folder, "name").text = "Aircraft Tracks"
        SubElement(tracks_folder, "open").text = "1"

        for aircraft in self.simulated_aircrafts:
            if aircraft.kml_track_data:
                description = (
                    f"Aircraft ID: {aircraft.aircraft_id}\n"
                    f"Type: {aircraft.aircraft_type if hasattr(aircraft, 'aircraft_type') else 'N/A'}\n"
                    f"Route: {aircraft.start_navpoint.name} -> {aircraft.end_navpoint.name}\n"
                    f"Speed: {aircraft.speed_kph} kph\n"
                    f"Altitude: FL{aircraft.altitude * 3.28084 / 100:.0f}"
                )
                
                # Determine style based on aircraft type (if EnhancedAircraft)
                style_id = "aircraftTrack" # Default style
                if isinstance(aircraft, EnhancedAircraft):
                    style_id = f"{aircraft.aircraft_type.lower()}_aircraft_style"

                tracks_folder.append(ElementTree.fromstring(_generate_kml_gxtrack(
                    f"{aircraft.aircraft_id} Track", description, aircraft.kml_track_data, style_id
                )))

        kml_content.append(ElementTree.tostring(tracks_folder, encoding='unicode', method='xml'))
        kml_content.append(_generate_kml_footer())

        file_path = filedialog.asksaveasfilename(
            defaultextension=".kml",
            filetypes=[("KML files", "*.kml")],
            title="Save Simulated Aircraft Tracks KML"
        )
        if file_path:
            try:
                xml_string = "\n".join(kml_content)
                dom = minidom.parseString(xml_string)
                pretty_xml_as_string = dom.toprettyxml(indent="  ")

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(pretty_xml_as_string)
                messagebox.showinfo("KML Export Success", f"Simulated Aircraft Tracks KML saved to:\n{file_path}")
                self.open_kml_in_google_earth(file_path)
            except Exception as e:
                messagebox.showerror("KML Export Error", f"Failed to save KML file: {e}")

    def generate_kml_sim_weather(self):
        """Generates a KML file with the current weather zones."""
        if not self.weather_zones:
            messagebox.showwarning("No Weather Zones", "No weather zones to export. Generate weather first.")
            return

        kml_content = []
        kml_content.append(_generate_kml_header("Simulated Weather Zones"))

        weather_folder = SubElement(Element("Folder"), "Folder")
        SubElement(weather_folder, "name").text = "Weather Zones"
        SubElement(weather_folder, "open").text = "1"

        for zone in self.weather_zones:
            description = (
                f"Zone Name: {zone.name}\n"
                f"Severity: {zone.severity.capitalize()}\n"
                f"Speed: {zone.speed_kph} kph\n"
                f"Direction: {math.degrees(zone.direction_rad):.1f}°"
            )
            coordinates = zone.vertices # Already includes closing point

            style_id = f"weatherZone{zone.severity.capitalize()}" # e.g., "weatherZoneSevere"

            weather_folder.append(ElementTree.fromstring(_generate_kml_polygon(
                zone.name, description, coordinates, style_url=style_id
            )))

        kml_content.append(ElementTree.tostring(weather_folder, encoding='unicode', method='xml'))
        kml_content.append(_generate_kml_footer())

        file_path = filedialog.asksaveasfilename(
            defaultextension=".kml",
            filetypes=[("KML files", "*.kml")],
            title="Save Simulated Weather Zones KML"
        )
        if file_path:
            try:
                xml_string = "\n".join(kml_content)
                dom = minidom.parseString(xml_string)
                pretty_xml_as_string = dom.toprettyxml(indent="  ")

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(pretty_xml_as_string)
                messagebox.showinfo("KML Export Success", f"Simulated Weather Zones KML saved to:\n{file_path}")
                self.open_kml_in_google_earth(file_path)
            except Exception as e:
                messagebox.showerror("KML Export Error", f"Failed to save KML file: {e}")

    def on_tab_change(self, event):
        """Callback for when the notebook tab changes."""
        selected_tab = self.notebook.tab(self.notebook.select(), "text")
        if selected_tab == "Advanced Features":
            # Ensure the airspace is plotted before starting simulation
            if self.current_airspace:
                self.show_complete_airspace() # Re-plot base airspace
                self.last_update_time = time.time() # Reset timer on tab change
            else:
                self.airspace_figure.clear()
                ax = self.airspace_figure.add_subplot(111)
                ax.text(0.5, 0.5, "Load an Airspace to use Advanced Features", ha='center', va='center', fontsize=12)
                ax.axis('off')
                self.airspace_canvas.draw()
        else:
            self.stop_traffic_simulation() # Stop simulation if not on advanced tab
            # Clear advanced plot elements when leaving the tab
            self.clear_aircraft_simulation() # This will clear aircraft artists
            self.clear_weather_zones() # This will clear weather artists
            # No need to explicitly redraw base airspace if switching to Airspace Analysis tab
            # as that tab's logic will call show_complete_airspace if needed.
            # If switching to Graph Editor, it will plot its own graph.
            
    def setup_visualization_area(self, parent, viz_type):
        """Setup visualization area for a tab with bigger graphs"""
        # Create bottom frame with visualization area and info panel
        bottom_frame = ttk.Frame(parent)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Graph visualization area - bigger and with cool styling
        viz_frame = ttk.LabelFrame(bottom_frame, text=f"{viz_type.title()} Visualization")
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure and canvas - increased size
        if viz_type == "airspace" or viz_type == "advanced": 
            # Use the same figure for airspace and advanced for layering
            # This logic ensures the figure is created once and reused
            if not hasattr(self, 'airspace_figure') or self.airspace_figure is None:
                self.airspace_figure = plt.Figure(figsize=(14, 10), dpi=100, facecolor=self.bg_very_light)
                self.airspace_figure.patch.set_facecolor(self.bg_very_light)
                
                self.airspace_canvas = FigureCanvasTkAgg(self.airspace_figure, viz_frame)
                self.airspace_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                
                toolbar_frame = ttk.Frame(viz_frame)
                toolbar_frame.pack(fill=tk.X, padx=5, pady=2)
                self.airspace_toolbar = NavigationToolbar2Tk(self.airspace_canvas, toolbar_frame)
                self.airspace_toolbar.update()
            else:
                # If figure already exists, ensure its widget is packed into the current viz_frame
                # This handles switching tabs, ensuring the canvas is visible in the active tab
                self.airspace_canvas.get_tk_widget().pack_forget() # Remove from previous parent
                self.airspace_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        else: # For graph tab
            self.graph_figure = plt.Figure(figsize=(14, 10), dpi=100, facecolor=self.bg_very_light)
            self.graph_figure.patch.set_facecolor(self.bg_very_light)
            
            self.graph_canvas = FigureCanvasTkAgg(self.graph_figure, viz_frame)
            self.graph_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            toolbar_frame = ttk.Frame(viz_frame)
            toolbar_frame.pack(fill=tk.X, padx=5, pady=2)
            self.graph_toolbar = NavigationToolbar2Tk(self.graph_canvas, toolbar_frame)
            self.graph_toolbar.update()
        
        # Info panel at the bottom with cool styling
        info_frame = ttk.LabelFrame(bottom_frame, text="Information")
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        if viz_type == "airspace":
            self.airspace_info_text = tk.Text(info_frame, height=6, wrap=tk.WORD,
                                            bg=self.bg_very_light, fg=self.text_dark,
                                            font=("Consolas", 9), relief="flat",
                                            borderwidth=1, highlightthickness=1,
                                            highlightcolor=self.accent_cyan)
            self.airspace_info_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        elif viz_type == "advanced": # Separate info text for advanced features
            self.advanced_info_text = tk.Text(info_frame, height=6, wrap=tk.WORD,
                                            bg=self.bg_very_light, fg=self.text_dark,
                                            font=("Consolas", 9), relief="flat",
                                            borderwidth=1, highlightthickness=1,
                                            highlightcolor=self.accent_cyan)
            self.advanced_info_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        else:
            self.graph_info_text = tk.Text(info_frame, height=6, wrap=tk.WORD,
                                         bg=self.bg_very_light, fg=self.text_dark,
                                         font=("Consolas", 9), relief="flat",
                                         borderwidth=1, highlightthickness=1,
                                         highlightcolor=self.accent_cyan)
            self.graph_info_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    # ===== NEW ENHANCED METHODS FOR 3D ZONES AND ENHANCED AIRCRAFT =====
    
    def setup_3d_airspace_zones(self):
        """Initialize 3D airspace zones for the current airspace"""
        if not self.current_airspace:
            messagebox.showwarning("No Airspace", "Load an airspace first to setup 3D zones.")
            return
            
        self.airspace_zones_3d = []
        
        # Get airspace bounds
        lats = [np.latitude for np in self.current_airspace.navpoints.values()]
        lons = [np.longitude for np in self.current_airspace.navpoints.values()]
        
        if not lats or not lons:
            return
            
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        # Create sample 3D airspace zones
        zones_config = [
            {
                'zone_id': 'TMA001',
                'name': 'Terminal Maneuvering Area Alpha',
                'zone_type': 'TMA',
                'vertices': [
                    (min_lon + 0.1, min_lat + 0.1),
                    (min_lon + 0.5, min_lat + 0.1),
                    (min_lon + 0.5, min_lat + 0.4),
                    (min_lon + 0.1, min_lat + 0.4)
                ],
                'min_alt': 5000,
                'max_alt': 15000,
                'description': 'High traffic terminal area'
            },
            {
                'zone_id': 'CTR001', 
                'name': 'Control Zone Bravo',
                'zone_type': 'CTR',
                'vertices': [
                    (max_lon - 0.3, max_lat - 0.3),
                    (max_lon - 0.1, max_lat - 0.3),
                    (max_lon - 0.1, max_lat - 0.1),
                    (max_lon - 0.3, max_lat - 0.1)
                ],
                'min_alt': 0,
                'max_alt': 10000,
                'description': 'Airport control zone'
            },
            {
                'zone_id': 'R001',
                'name': 'Restricted Area R-2508',
                'zone_type': 'RESTRICTED',
                'vertices': [
                    ((min_lon + max_lon) / 2 - 0.15, (min_lat + max_lat) / 2 - 0.1),
                    ((min_lon + max_lon) / 2 + 0.15, (min_lat + max_lat) / 2 - 0.1),
                    ((min_lon + max_lon) / 2 + 0.15, (min_lat + max_lat) / 2 + 0.1),
                    ((min_lon + max_lon) / 2 - 0.15, (min_lat + max_lat) / 2 + 0.1)
                ],
                'min_alt': 0,
                'max_alt': 60000,
                'description': '🚨 MILITARY OPERATIONS AREA - FIREJET INTERCEPT ACTIVE'
            }
        ]
        
        for config in zones_config:
            zone = AirspaceZone3D(
                config['zone_id'], config['name'], config['zone_type'],
                config['vertices'], config['min_alt'], config['max_alt'],
                config['description']
            )
            self.airspace_zones_3d.append(zone)
            
        if hasattr(self, 'advanced_info_text'):
            self.advanced_info_text.insert(tk.END, f"✅ Created {len(self.airspace_zones_3d)} 3D airspace zones\n")
        print(f"✅ Created {len(self.airspace_zones_3d)} 3D airspace zones")

    def spawn_enhanced_aircraft(self):
        """Spawn enhanced aircraft with random types"""
        if not self.current_airspace or not self.current_airspace.navpoints:
            messagebox.showwarning("No Airspace", "Load an airspace first to spawn aircraft.")
            return
            
        # Initialize 3D zones if not already done
        if not hasattr(self, 'airspace_zones_3d') or not self.airspace_zones_3d:
            self.setup_3d_airspace_zones()

        nav_points_list = list(self.current_airspace.navpoints.values())
        if len(nav_points_list) < 2:
            messagebox.showwarning("Not Enough Points", "Need at least two navigation points.")
            return

        # Select random aircraft type with weighted probabilities
        aircraft_types = ['AIRLINER', 'BUSINESS_JET', 'FIGHTER_JET', 'HELICOPTER']
        weights = [0.4, 0.3, 0.2, 0.1]  # More airliners, fewer helicopters
        aircraft_type = random.choices(aircraft_types, weights=weights)[0]

        origin_point = random.choice(nav_points_list)
        destination_point = random.choice(nav_points_list)
        while origin_point == destination_point:
            destination_point = random.choice(nav_points_list)

        aircraft_id = f"{aircraft_type[:2]}{len(self.simulated_aircrafts) + 1:03d}"
        new_aircraft = EnhancedAircraft(aircraft_id, origin_point, destination_point, aircraft_type)
        
        if new_aircraft.calculate_path(self.current_airspace, destination_point):
            self.simulated_aircrafts.append(new_aircraft)
            if hasattr(self, 'advanced_info_text'):
                self.advanced_info_text.insert(tk.END, 
                    f"✈️ Spawned {aircraft_type} {aircraft_id}: {origin_point.name} -> {destination_point.name} "
                    f"(FL{new_aircraft.altitude * 3.28084 / 100:.0f}, {new_aircraft.speed_kph} kph)\n")
            self.redraw_animated_elements()
        else:
            if hasattr(self, 'advanced_info_text'):
                self.advanced_info_text.insert(tk.END, f"❌ Failed to spawn {aircraft_type} {aircraft_id}\n")

    def monitor_3d_airspace_violations(self):
        """Monitor aircraft for 3D airspace zone violations and trigger effects"""
        if not hasattr(self, 'airspace_zones_3d') or not self.airspace_zones_3d:
            return
            
        for aircraft in self.simulated_aircrafts:
            if not isinstance(aircraft, EnhancedAircraft):
                continue
                
            # Update aircraft scale based on altitude
            aircraft.update_scale_by_altitude()
            
            for zone in self.airspace_zones_3d:
                was_inside = aircraft.aircraft_id in zone.aircraft_inside
                is_inside = zone.contains_aircraft(aircraft)
                
                if is_inside and not was_inside:
                    # Aircraft just entered zone
                    zone.aircraft_inside.add(aircraft.aircraft_id)
                    zone.trigger_flash_effect()
                    
                    if zone.zone_type in ['RESTRICTED', 'PROHIBITED']:
                        aircraft.activate_firejet_warning()
                        if hasattr(self, 'advanced_info_text'):
                            self.advanced_info_text.insert(tk.END, 
                                f"🚨 FIREJET SCRAMBLED: {aircraft.aircraft_id} violated {zone.name}!\n")
                        
                        # Trigger emergency rerouting
                        self.emergency_reroute_aircraft(aircraft, zone)
                        
                    if hasattr(self, 'advanced_info_text'):
                        self.advanced_info_text.insert(tk.END, 
                            f"⚠️ {aircraft.aircraft_id} entered {zone.name} ({zone.zone_type})\n")
                        
                elif not is_inside and was_inside:
                    # Aircraft left zone
                    zone.aircraft_inside.discard(aircraft.aircraft_id)
                    if zone.zone_type in ['RESTRICTED', 'PROHIBITED']:
                        aircraft.deactivate_firejet_warning()
                        if hasattr(self, 'advanced_info_text'):
                            self.advanced_info_text.insert(tk.END, 
                                f"✅ {aircraft.aircraft_id} cleared restricted airspace\n")
                
                # Update flash states
                zone.update_flash_state()

    def emergency_reroute_aircraft(self, aircraft, restricted_zone):
        """Emergency rerouting when aircraft violates restricted airspace"""
        aircraft.emergency_reroute_attempts += 1
        
        if aircraft.emergency_reroute_attempts > 3:
            # Force landing after 3 attempts
            if hasattr(self, 'advanced_info_text'):
                self.advanced_info_text.insert(tk.END, 
                    f"🛑 EMERGENCY LANDING: {aircraft.aircraft_id} forced to land!\n")
            aircraft.is_active = False
            return
        
        # Create exclusion zones around all restricted areas
        excluded_points = set()
        excluded_segments = set()
        
        for zone in self.airspace_zones_3d:
            if zone.zone_type in ['RESTRICTED', 'PROHIBITED']:
                # Exclude all navpoints within restricted zones
                for np_id, nav_point in self.current_airspace.navpoints.items():
                    if zone._point_in_polygon(nav_point.longitude, nav_point.latitude):
                        excluded_points.add(np_id)
                
                # Exclude segments crossing restricted zones
                for segment in self.current_airspace.navsegments:
                    origin_np = self.current_airspace.navpoints.get(segment.origin_number)
                    dest_np = self.current_airspace.navpoints.get(segment.destination_number)
                    if origin_np and dest_np:
                        mid_lon = (origin_np.longitude + dest_np.longitude) / 2
                        mid_lat = (origin_np.latitude + dest_np.latitude) / 2
                        if zone._point_in_polygon(mid_lon, mid_lat):
                            excluded_segments.add((origin_np.number, dest_np.number))
        
        # Attempt emergency rerouting
        original_destination = aircraft.path_to_destination[-1] if aircraft.path_to_destination else aircraft.end_navpoint
        if aircraft.calculate_path(self.current_airspace, original_destination, excluded_points, excluded_segments):
            aircraft.rerouting = True
            if hasattr(self, 'advanced_info_text'):
                self.advanced_info_text.insert(tk.END, 
                    f"🔄 EMERGENCY REROUTE: {aircraft.aircraft_id} attempting alternate path\n")
        else:
            # No safe path found - emergency descent
            if hasattr(self, 'advanced_info_text'):
                self.advanced_info_text.insert(tk.END, 
                    f"⬇️ EMERGENCY DESCENT: {aircraft.aircraft_id} no alternate route found\n")
            aircraft.altitude = max(1000, aircraft.altitude - 5000)  # Emergency descent

    def generate_kml_3d_airspace_zones(self):
        """Generate KML file with 3D airspace zones"""
        if not hasattr(self, 'airspace_zones_3d') or not self.airspace_zones_3d:
            messagebox.showwarning("No 3D Zones", "No 3D airspace zones to export. Setup zones first.")
            return
            
        kml_content = []
        kml_content.append(_generate_kml_header("3D Airspace Zones"))
        
        # Add styles for different zone types
        for zone_type, props in AirspaceZone3D.ZONE_TYPES.items():
            zone_dummy = AirspaceZone3D(None, None, zone_type, [], 0, 0)
            kml_color = zone_dummy._rgb_to_kml_color(props['color'], props['alpha'])
            style = f"""
    <Style id="{zone_type.lower()}_zone_style">
        <LineStyle>
            <color>{kml_color}</color>
            <width>3</width>
        </LineStyle>
        <PolyStyle>
            <color>{kml_color}</color>
            <fill>1</fill>
            <outline>1</outline>
        </PolyStyle>
    </Style>"""
            kml_content.append(style)
        
        # Add 3D polygons for each zone
        for zone in self.airspace_zones_3d:
            kml_content.append(zone.get_kml_polygon_3d())
        
        kml_content.append(_generate_kml_footer())
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".kml",
            filetypes=[("KML files", "*.kml")],
            title="Save 3D Airspace Zones KML"
        )
        if file_path:
            try:
                xml_string = "\n".join(kml_content)
                dom = minidom.parseString(xml_string)
                pretty_xml_as_string = dom.toprettyxml(indent="  ")

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(pretty_xml_as_string)
                messagebox.showinfo("KML Export Success", f"3D Airspace Zones KML saved to:\n{file_path}")
                self.open_kml_in_google_earth(file_path)
            except Exception as e:
                messagebox.showerror("KML Export Error", f"Failed to save KML file: {e}")

    def generate_kml_enhanced_aircraft_models(self):
        """Generate KML file with 3D aircraft models"""
        if not self.simulated_aircrafts:
            messagebox.showwarning("No Aircraft", "No simulated aircraft to export.")
            return
            
        kml_content = []
        kml_content.append(_generate_kml_header("Enhanced Aircraft 3D Models"))
        
        # Add 3D models for each aircraft
        for aircraft in self.simulated_aircrafts:
            if isinstance(aircraft, EnhancedAircraft):
                kml_content.append(aircraft.get_kml_model_3d())
        
        kml_content.append(_generate_kml_footer())
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".kml",
            filetypes=[("KML files", "*.kml")],
            title="Save Enhanced Aircraft Models KML"
        )
        if file_path:
            try:
                xml_string = "\n".join(kml_content)
                dom = minidom.parseString(xml_string)
                pretty_xml_as_string = dom.toprettyxml(indent="  ")

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(pretty_xml_as_string)
                messagebox.showinfo("KML Export Success", f"Enhanced Aircraft Models KML saved to:\n{file_path}")
                self.open_kml_in_google_earth(file_path)
            except Exception as e:
                messagebox.showerror("KML Export Error", f"Failed to save KML file: {e}")

    # ===== ENHANCED TRAFFIC SIMULATION (Updated to include 3D zone monitoring) =====
    
    def update_traffic_simulation(self):
        """Enhanced traffic simulation update with 3D zone monitoring"""
        if not self.simulation_active:
            return

        current_time_epoch = time.time()
        delta_time_seconds = (current_time_epoch - self.last_update_time) * self.simulation_speed.get()
        delta_time_hours = delta_time_seconds / 3600.0
        self.last_update_time = current_time_epoch

        # Update aircraft positions
        active_aircrafts_next_step = []
        for aircraft in self.simulated_aircrafts:
            if aircraft.is_active:
                aircraft.update_position(delta_time_hours, current_time_epoch)
                if aircraft.is_active:
                    active_aircrafts_next_step.append(aircraft)
                else:
                    if hasattr(self, 'advanced_info_text'):
                        self.advanced_info_text.insert(tk.END, f"Aircraft {aircraft.aircraft_id} arrived.\n")
        self.simulated_aircrafts = active_aircrafts_next_step

        # Update weather zones
        for weather_zone in self.weather_zones:
            weather_zone.update_position(delta_time_hours)

        # ENHANCED: Monitor 3D airspace violations and firejet warnings
        self.monitor_3d_airspace_violations()

        # Perform dynamic rerouting
        self.dynamic_reroute_for_weather()
        
        # Detect conflicts
        self.detect_conflicts()
        
        # Redraw all animated elements
        self.redraw_animated_elements()

        # Schedule next update
        self.animation_job = self.root.after(self.animation_interval_ms, self.update_traffic_simulation)

# Part 4: Airspace Functions, KML Export Functions, and Google Earth Integration (Continued)

    # ===== Airspace Functions (Mostly Unchanged, with KML hooks) =====
    def load_airspace(self, airspace_name):
        """Load data for the specified airspace"""
        try:
            # File prefixes based on airspace
            prefix_map = {
                "Catalunya": "Cat_",
                "España": "Esp_",
                "Europe": "Eur_"
            }
            prefix = prefix_map.get(airspace_name, "Cat_")
            
            # Get file directory if not already loaded
            if not self.airspaces[airspace_name]:
                # Ask user to select the specific airspace folder directly
                file_dir = filedialog.askdirectory(
                    title=f"Select the {airspace_name.upper()} folder (containing {prefix}nav.txt, {prefix}seg.txt, {prefix}aer.txt)"
                )
                if not file_dir:
                    return
                
                # The selected folder should contain the files directly
                nav_file = os.path.join(file_dir, f"{prefix}nav.txt")
                seg_file = os.path.join(file_dir, f"{prefix}seg.txt")
                aer_file = os.path.join(file_dir, f"{prefix}aer.txt")
                
                # Check if files exist in the selected folder
                missing_files = []
                existing_files = []
                
                for file_path, file_name in [(nav_file, f"{prefix}nav.txt"), 
                                           (seg_file, f"{prefix}seg.txt"), 
                                           (aer_file, f"{prefix}aer.txt")]:
                    if os.path.exists(file_path):
                        existing_files.append(file_name)
                    else:
                        missing_files.append(file_name)
                
                # Show what we found
                print(f"\n{'='*60}")
                print(f"CHECKING FILES IN: {file_dir}")
                print(f"{'='*60}")
                print(f"Looking for: {prefix}nav.txt, {prefix}seg.txt, {prefix}aer.txt")
                print(f"Found: {existing_files}")
                if missing_files:
                    print(f"Missing: {missing_files}")
                
                # List all files in the directory for debugging
                try:
                    all_files = os.listdir(file_dir)
                    print(f"All files in directory: {all_files}")
                except Exception as e:
                    print(f"Could not list directory contents: {e}")
                
                if missing_files:
                    messagebox.showerror("Missing Files", 
                        f"Could not find all required files in:\n{file_dir}\n\n"
                        f"Found: {', '.join(existing_files) if existing_files else 'None'}\n"
                        f"Missing: {', '.join(missing_files)}\n\n"
                        f"Please make sure you select the folder that contains:\n"
                        f"• {prefix}nav.txt\n"
                        f"• {prefix}seg.txt\n"
                        f"• {prefix}aer.txt")
                    return
                
                # Load the data
                print(f"\n{'='*60}")
                print(f"LOADING {airspace_name.upper()} AIRSPACE")
                print(f"{'='*60}")
                print(f"Files location: {file_dir}")
                
                # Create airspace
                self.airspaces[airspace_name] = AirSpace(airspace_name)
                
                # Load navpoints first
                print(f"\nLoading navigation points from: {nav_file}")
                navpoints = parse_navpoints_file(nav_file)
                print(f"✓ Parsed {len(navpoints)} navigation points")
                
                if len(navpoints) == 0:
                    print("⚠ Warning: No navigation points were parsed!")
                    # Show first few lines of the file for debugging
                    try:
                        with open(nav_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()[:5]
                            print("First 5 lines of navigation file:")
                            for i, line in enumerate(lines, 1):
                                print(f"  Line {i}: '{line.strip()}'")
                    except Exception as e:
                        print(f"Could not read nav file: {e}")
                
                for nav_id, navpoint in navpoints.items():
                    self.airspaces[airspace_name].add_navpoint(navpoint)
                
                # Load segments
                print(f"\nLoading segments from: {seg_file}")
                segments = parse_segments_file(seg_file)
                print(f"✓ Parsed {len(segments)} segments")
                
                if len(segments) == 0:
                    print("⚠ Warning: No segments were parsed!")
                    try:
                        with open(seg_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()[:5]
                            print("First 5 lines of segments file:")
                            for i, line in enumerate(lines, 1):
                                print(f"  Line {i}: '{line.strip()}'")
                    except Exception as e:
                        print(f"Could not read seg file: {e}")
                
                for segment in segments:
                    self.airspaces[airspace_name].add_segment(segment)
                
                # Load airports
                print(f"\nLoading airports from: {aer_file}")
                airports = parse_airports_file(aer_file, navpoints)
                print(f"✓ Parsed {len(airports)} airports")
                
                if len(airports) == 0:
                    print("⚠ Warning: No airports were parsed!")
                    try:
                        with open(aer_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()[:10]
                            print("First 10 lines of airports file:")
                            for i, line in enumerate(lines, 1):
                                print(f"  Line {i}: '{line.strip()}'")
                    except Exception as e:
                        print(f"Could not read aer file: {e}")
                
                for airport_name, airport in airports.items():
                    self.airspaces[airspace_name].add_airport(airport)
                
                # Final verification
                final_navpoints = len(self.airspaces[airspace_name].navpoints)
                final_segments = len(self.airspaces[airspace_name].navsegments)
                final_airports = len(self.airspaces[airspace_name].navairports)
                
                print(f"\n{'='*60}")
                print(f"LOADING COMPLETE FOR {airspace_name.upper()}")
                print(f"{'='*60}")
                print(f"Navigation Points: {final_navpoints}")
                print(f"Segments: {final_segments}")
                print(f"Airports: {final_airports}")
                
                if final_navpoints > 0:
                    messagebox.showinfo("Success", 
                        f"{airspace_name} loaded successfully!\n\n"
                        f"Navigation Points: {final_navpoints}\n"
                        f"Segments: {final_segments}\n"
                        f"Airports: {final_airports}")
                else:
                    messagebox.showerror("Loading Failed", 
                        f"No navigation points were loaded for {airspace_name}.\n"
                        f"Please check the console output for details.\n"
                        f"The file format might not match the expected format.")
                    return
            
            # Set as current airspace
            self.current_airspace = self.airspaces[airspace_name]
            self.current_airspace_name.set(airspace_name)
            
            # Update UI elements
            self.update_airspace_ui_elements()
            
            # Show the airspace
            self.show_complete_airspace()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading {airspace_name}:\n{str(e)}")
            print(f"Error loading {airspace_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def update_airspace_ui_elements(self):
        """Update UI elements based on the current airspace"""
        if not self.current_airspace:
            return
            
        # Update node selection dropdowns
        nav_points = [nav_point.name for nav_point in self.current_airspace.navpoints.values()]
        self.source_combo['values'] = sorted(nav_points)
        self.dest_combo['values'] = sorted(nav_points)
        
        # Update airport selection dropdown
        airports = list(self.current_airspace.navairports.keys())
        self.airport_combo['values'] = sorted(airports)
    
    def show_complete_airspace(self):
        """Display the complete airspace with clean arrows, distances, and node names"""
        if not self.current_airspace:
            # Clear current plot if no airspace is loaded
            self.airspace_figure.clear()
            ax = self.airspace_figure.add_subplot(111)
            ax.text(0.5, 0.5, "No Airspace Loaded", ha='center', va='center', fontsize=14, color=self.bg_secondary)
            ax.axis('off')
            self.airspace_canvas.draw()
            self.airspace_info_text.delete(1.0, tk.END)
            self.airspace_info_text.insert(tk.END, "No airspace loaded. Please load one from the 'Airspace Selection' section.\n")
            # Also clear advanced info if it's the current tab
            if hasattr(self, 'advanced_info_text') and self.notebook.tab(self.notebook.select(), "text") == "Advanced Features":
                 self.advanced_info_text.delete(1.0, tk.END)
                 self.advanced_info_text.insert(tk.END, "No airspace loaded. Please load one first.\n")
            return
        
        print(f"\n=== VISUALIZING {self.current_airspace_name.get().upper()} ===")
        
        self.airspace_figure.clear()
        ax = self.airspace_figure.add_subplot(111)
        
        # Set cool background color for the plot
        ax.set_facecolor('#f0f8ff')  # Alice blue background
        
        # Add cool grid styling
        ax.grid(True, linestyle='--', color=self.accent_cyan, alpha=0.4, linewidth=0.5)
        
        # Get coordinate ranges
        lats = [nav_point.latitude for nav_point in self.current_airspace.navpoints.values()]
        lons = [nav_point.longitude for nav_point in self.current_airspace.navpoints.values()]
        
        print(f"Coordinate ranges: Lat {min(lats):.2f} to {max(lats):.2f}, Lon {min(lons):.2f} to {max(lons):.2f}")
        
        # Plot segments with arrows and distances - cleaner version
        segments_plotted = 0
        for segment in self.current_airspace.navsegments:
            origin = self.current_airspace.navpoints.get(segment.origin_number)
            destination = self.current_airspace.navpoints.get(segment.destination_number)
            
            if origin and destination:
                # Draw arrow from origin to destination with cool colors
                ax.annotate('', xy=(destination.longitude, destination.latitude), 
                           xytext=(origin.longitude, origin.latitude),
                           arrowprops=dict(arrowstyle='->', color=self.accent_teal, lw=1.2, alpha=0.8))
                
                # Add distance labels - smaller and cleaner
                mid_lon = (origin.longitude + destination.longitude) / 2
                mid_lat = (origin.latitude + destination.latitude) / 2
                ax.text(mid_lon, mid_lat, f"{segment.distance:.1f}", 
                        color='#d53f8c', fontsize=5, fontweight='normal', alpha=0.8)  # Pink for visibility
                segments_plotted += 1
        
        # Plot navigation points with names - smaller and no background
        points_plotted = 0
        for nav_point in self.current_airspace.navpoints.values():
            ax.scatter(nav_point.longitude, nav_point.latitude, s=25, c=self.bg_primary, marker='o', zorder=2, alpha=0.9)
            # Add node names - smaller and cleaner
            ax.text(nav_point.longitude + 0.005, nav_point.latitude + 0.005, nav_point.name, 
                    fontsize=6, ha='left', va='bottom', color=self.bg_primary, alpha=0.9, fontweight='bold')
            points_plotted += 1
        
        # Plot airports with special markers - cleaner labels
        airports_plotted = 0
        for airport_name, airport in self.current_airspace.navairports.items():
            # Find a navpoint associated with the airport to plot its location
            airport_nav_point = None
            if airport.sids:
                airport_nav_point = self.current_airspace.navpoints.get(airport.sids[0])
            elif airport.stars:
                airport_nav_point = self.current_airspace.navpoints.get(airport.stars[0])

            if airport_nav_point:
                ax.scatter(airport_nav_point.longitude, airport_nav_point.latitude, s=120, c='#e53e3e', marker='^', zorder=3, alpha=0.9)
                ax.text(airport_nav_point.longitude, airport_nav_point.latitude + 0.01, airport_name, 
                           fontsize=8, ha='center', va='bottom', color='#e53e3e', fontweight='bold', alpha=0.9)
                airports_plotted += 1
        
        # Cool title and labels
        ax.set_title(f"🛩️ Airspace: {self.current_airspace_name.get()}", 
                    fontsize=16, color=self.bg_primary, fontweight='bold', pad=20)
        ax.set_xlabel("Longitude", fontsize=12, color=self.text_dark)
        ax.set_ylabel("Latitude", fontsize=12, color=self.text_dark)
        ax.set_aspect('equal', adjustable='box')
        
        # Style the axes with cool colors
        ax.spines['top'].set_color(self.accent_teal)
        ax.spines['bottom'].set_color(self.accent_teal)
        ax.spines['left'].set_color(self.accent_teal)
        ax.spines['right'].set_color(self.accent_teal)
        ax.tick_params(colors=self.text_dark)
        
        # Set proper limits
        if lats and lons:
            lat_padding = (max(lats) - min(lats)) * 0.05 if (max(lats) - min(lats)) > 0 else 0.1
            lon_padding = (max(lons) - min(lons)) * 0.05 if (max(lons) - min(lons)) > 0 else 0.1
            ax.set_xlim(min(lons) - lon_padding, max(lons) + lon_padding)
            ax.set_ylim(min(lats) - lat_padding, max(lats) + lat_padding)
        else: # Default limits if no points loaded
            ax.set_xlim(-10, 10)
            ax.set_ylim(30, 50) # Approx for Europe
        
        self.airspace_canvas.draw()
        
        # Update info with cool styling and emojis
        self.airspace_info_text.delete(1.0, tk.END)
        self.airspace_info_text.insert(tk.END, f"🌍 Airspace: {self.current_airspace_name.get()}\n")
        self.airspace_info_text.insert(tk.END, f"📍 Navigation Points: {len(self.current_airspace.navpoints)}\n")
        self.airspace_info_text.insert(tk.END, f"🔗 Segments: {len(self.current_airspace.navsegments)}\n")
        self.airspace_info_text.insert(tk.END, f"✈️ Airports: {len(self.current_airspace.navairports)}\n")
        self.airspace_info_text.insert(tk.END, f"🎨 Clean visualization with arrows, distances, and node names\n")
        
        print(f"Visualization complete: {points_plotted} points, {segments_plotted} segments, {airports_plotted} airports")
        self.airspace_ax = ax # Store the axes for later use in animation

    def get_navpoint_by_name(self, name):
        """Find a NavPoint by its name"""
        if not self.current_airspace:
            return None
        for nav_point in self.current_airspace.navpoints.values():
            if nav_point.name == name:
                return nav_point
        return None
    
    def show_neighbors(self):
        """Show neighbors of the selected node"""
        if not self.current_airspace:
            messagebox.showwarning("No Airspace", "Please load an airspace first.")
            return
        
        source_name = self.source_node.get()
        if not source_name:
            messagebox.showwarning("No Source", "Please select a source node.")
            return
        
        source_point = self.get_navpoint_by_name(source_name)
        if not source_point:
            messagebox.showerror("Error", f"Could not find source node: {source_name}")
            return
        
        neighbors = self.current_airspace.get_neighbors(source_point.number)
        
        self.airspace_figure.clear()
        ax = self.airspace_figure.add_subplot(111)
        ax.grid(True, linestyle='--', color='#ffcccc', alpha=0.6)
        
        # Plot all points (faded)
        for nav_point in self.current_airspace.navpoints.values():
            if nav_point != source_point and nav_point not in neighbors:
                ax.scatter(nav_point.longitude, nav_point.latitude, s=3, c='lightgray', marker='o')
        
        # Highlight source
        ax.scatter(source_point.longitude, source_point.latitude, s=50, c='red', marker='o')
        ax.annotate(source_point.name, (source_point.longitude, source_point.latitude), 
                   fontsize=8, ha='right', va='bottom', fontweight='bold')
        
        # Highlight neighbors
        for neighbor in neighbors:
            ax.scatter(neighbor.longitude, neighbor.latitude, s=30, c='green', marker='o')
            ax.annotate(neighbor.name, (neighbor.longitude, neighbor.latitude), 
                       fontsize=6, ha='right', va='bottom')
            
            # Draw connection
            ax.plot([source_point.longitude, neighbor.longitude], 
                   [source_point.latitude, neighbor.latitude], 
                   color='blue', linewidth=2)
        
        ax.set_title(f"Neighbors of {source_name}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect('equal', adjustable='box')
        
        self.airspace_canvas.draw()
        
        # Update info
        self.airspace_info_text.delete(1.0, tk.END)
        self.airspace_info_text.insert(tk.END, f"Source: {source_name}\n")
        self.airspace_info_text.insert(tk.END, f"Neighbors: {len(neighbors)}\n\n")
        for neighbor in neighbors:
            self.airspace_info_text.insert(tk.END, f"- {neighbor.name}\n")
    
    def show_reachability(self):
        """Show reachable nodes"""
        if not self.current_airspace:
            messagebox.showwarning("No Airspace", "Please load an airspace first.")
            return
        
        source_name = self.source_node.get()
        if not source_name:
            messagebox.showwarning("No Source", "Please select a source node.")
            return
        
        source_point = self.get_navpoint_by_name(source_name)
        if not source_point:
            messagebox.showerror("Error", f"Could not find source node: {source_name}")
            return
        
        reachable_ids = self.current_airspace.find_reachable_points(source_point.number)
        
        self.airspace_figure.clear()
        ax = self.airspace_figure.add_subplot(111)
        ax.grid(True, linestyle='--', color='#ffcccc', alpha=0.6)
        
        # Plot all points
        for nav_point in self.current_airspace.navpoints.values():
            if nav_point.number in reachable_ids:
                color = 'red' if nav_point.number == source_point.number else 'green'
                size = 50 if nav_point.number == source_point.number else 15
                ax.scatter(nav_point.longitude, nav_point.latitude, s=size, c=color, marker='o')
            else:
                ax.scatter(nav_point.longitude, nav_point.latitude, s=3, c='lightgray', marker='o')
        
        ax.set_title(f"Points Reachable from {source_name}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect('equal', adjustable='box')
        
        self.airspace_canvas.draw()
        
        # Update info
        self.airspace_info_text.delete(1.0, tk.END)
        self.airspace_info_text.insert(tk.END, f"Source: {source_name}\n")
        self.airspace_info_text.insert(tk.END, f"Reachable points: {len(reachable_ids)}\n")
        self.airspace_info_text.insert(tk.END, f"Total points: {len(self.current_airspace.navpoints)}\n")
    
    def show_shortest_path(self):
        """Show shortest path between two nodes"""
        if not self.current_airspace:
            messagebox.showwarning("No Airspace", "Please load an airspace first.")
            return
        
        source_name = self.source_node.get()
        dest_name = self.dest_node.get()
        
        if not source_name or not dest_name:
            messagebox.showwarning("Missing Selection", "Please select both source and destination nodes.")
            return
        
        if source_name == dest_name:
            messagebox.showwarning("Same Node", "Source and destination cannot be the same.")
            return
        
        source_point = self.get_navpoint_by_name(source_name)
        dest_point = self.get_navpoint_by_name(dest_name)
        
        if not source_point or not dest_point:
            messagebox.showerror("Error", "Could not find one or both nodes.")
            return
        
        path, cost = self.current_airspace.find_shortest_path(source_point.number, dest_point.number)
        
        if not path or cost == float('inf'):
            messagebox.showinfo("No Path", f"No path exists from {source_name} to {dest_name}.")
            self.current_display_path_nodes = [] # Clear stored path if no path found
            self.current_display_path_cost = float('inf')
            return
        
        self.current_display_path_nodes = [self.current_airspace.navpoints[node_id] for node_id in path]
        self.current_display_path_cost = cost

        self.airspace_figure.clear()
        ax = self.airspace_figure.add_subplot(111)
        ax.grid(True, linestyle='--', color='#ffcccc', alpha=0.6)
        
        # Plot all points (faded)
        for nav_point in self.current_airspace.navpoints.values():
            if nav_point.number not in path:
                ax.scatter(nav_point.longitude, nav_point.latitude, s=3, c='lightgray', marker='o')
        
        # Plot path
        path_points = [self.current_airspace.navpoints[node_id] for node_id in path]
        
        # Draw path segments
        for i in range(len(path_points)-1):
            start = path_points[i]
            end = path_points[i+1]
            ax.plot([start.longitude, end.longitude], [start.latitude, end.latitude], 
                    color='blue', linewidth=3)
        
        # Plot path nodes
        for i, point in enumerate(path_points):
            if i == 0:  # Source
                ax.scatter(point.longitude, point.latitude, s=50, c='green', marker='o')
                ax.annotate(point.name, (point.longitude, point.latitude), 
                           fontsize=8, ha='right', va='bottom', fontweight='bold', color='green')
            elif i == len(path_points)-1:  # Destination
                ax.scatter(point.longitude, point.latitude, s=50, c='red', marker='o')
                ax.annotate(point.name, (point.longitude, point.latitude), 
                           fontsize=8, ha='right', va='bottom', fontweight='bold', color='red')
            else:  # Intermediate points
                ax.scatter(point.longitude, point.latitude, s=20, c='blue', marker='o')
        
        ax.set_title(f"Shortest Path: {source_name} → {dest_name} (Cost: {cost:.2f} km)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect('equal', adjustable='box')
        
        self.airspace_canvas.draw()
        
        # Update info
        self.airspace_info_text.delete(1.0, tk.END)
        self.airspace_info_text.insert(tk.END, f"Path: {source_name} → {dest_name}\n")
        self.airspace_info_text.insert(tk.END, f"Total Distance: {cost:.2f} km\n")
        self.airspace_info_text.insert(tk.END, f"Segments: {len(path)-1}\n\n")
        self.airspace_info_text.insert(tk.END, "Route:\n")
        for i in range(len(path_points)-1):
            segment = self.current_airspace.get_segment(path_points[i].number, path_points[i+1].number)
            distance = segment.distance if segment else 0
            self.airspace_info_text.insert(tk.END, f"{path_points[i].name} → {path_points[i+1].name} ({distance:.1f} km)\n")
    
    def show_airport_info(self):
        """Show airport information"""
        if not self.current_airspace:
            messagebox.showwarning("No Airspace", "Please load an airspace first.")
            return
        
        airport_name = self.airport.get()
        if not airport_name:
            messagebox.showwarning("No Airport", "Please select an airport.")
            return
        
        airport = self.current_airspace.navairports.get(airport_name)
        if not airport:
            messagebox.showerror("Error", f"Could not find airport: {airport_name}")
            return
        
        self.airspace_figure.clear()
        ax = self.airspace_figure.add_subplot(111)
        ax.grid(True, linestyle='--', color='#ffcccc', alpha=0.6)
        
        # Plot all points (faded)
        for nav_point in self.current_airspace.navpoints.values():
            ax.scatter(nav_point.longitude, nav_point.latitude, s=3, c='lightgray', marker='o')
        
        # Plot airport location (use first SID or STAR navpoint for coords)
        airport_point = None
        if airport.sids:
            airport_point = self.current_airspace.navpoints.get(airport.sids[0])
        if not airport_point and airport.stars: # If no SID point, try STAR
            airport_point = self.current_airspace.navpoints.get(airport.stars[0])

        if airport_point:
            ax.scatter(airport_point.longitude, airport_point.latitude, s=200, c='red', marker='^')
            ax.annotate(airport_name, (airport_point.longitude, airport_point.latitude), 
                       fontsize=12, ha='center', va='bottom', fontweight='bold', color='red')
        
        # Plot SIDs
        for sid_id in airport.sids:
            sid_point = self.current_airspace.navpoints.get(sid_id)
            if sid_point:
                ax.scatter(sid_point.longitude, sid_point.latitude, s=50, c='green', marker='o')
                ax.annotate(sid_point.name, (sid_point.longitude, sid_point.latitude), 
                           fontsize=6, ha='right', va='bottom', color='green')
        
        # Plot STARs
        for star_id in airport.stars:
            star_point = self.current_airspace.navpoints.get(star_id)
            if star_point:
                ax.scatter(star_point.longitude, star_point.latitude, s=50, c='blue', marker='o')
                ax.annotate(star_point.name, (star_point.longitude, star_point.latitude), 
                           fontsize=6, ha='right', va='bottom', color='blue')
        
        ax.set_title(f"Airport: {airport_name}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect('equal', adjustable='box')
        
        # Add legend
        ax.scatter([], [], s=50, c='green', marker='o', label='SIDs (Departures)')
        ax.scatter([], [], s=50, c='blue', marker='o', label='STARs (Arrivals)')
        ax.scatter([], [], s=200, c='red', marker='^', label='Airport')
        ax.legend()
        
        self.airspace_canvas.draw()
        
        # Update info
        self.airspace_info_text.delete(1.0, tk.END)
        self.airspace_info_text.insert(tk.END, f"Airport: {airport_name}\n\n")
        self.airspace_info_text.insert(tk.END, f"SIDs (Departures): {len(airport.sids)}\n")
        for i, sid_id in enumerate(airport.sids):
            sid_point = self.current_airspace.navpoints.get(sid_id)
            if sid_point:
                self.airspace_info_text.insert(tk.END, f"  {i+1}. {sid_point.name}\n")
        
        self.airspace_info_text.insert(tk.END, f"\nSTARs (Arrivals): {len(airport.stars)}\n")
        for i, star_id in enumerate(airport.stars):
            star_point = self.current_airspace.navpoints.get(star_id)
            if star_point:
                self.airspace_info_text.insert(tk.END, f"  {i+1}. {star_point.name}\n")

    # ===== KML Export Functions (New as per your request) =====
    def generate_kml_airspace(self):
        """Generates a KML file for the entire loaded airspace."""
        if not self.current_airspace:
            messagebox.showwarning("No Airspace", "Please load an airspace first to export to KML.")
            return

        kml_content = []
        kml_content.append(_generate_kml_header(f"{self.current_airspace.name} Airspace"))

        # Add Navigation Points
        for np_id, nav_point in self.current_airspace.navpoints.items():
            description = f"NavPoint: {nav_point.name}\nID: {nav_point.number}\nLat: {nav_point.latitude}, Lon: {nav_point.longitude}"
            kml_content.append(_generate_kml_placemark_point(
                nav_point.name, description, nav_point.longitude, nav_point.latitude, style_url="navpointIcon"
            ))
        
        # Add Airports (using the first SID/STAR point for location)
        for airport_name, airport in self.current_airspace.navairports.items():
            airport_point = None
            if airport.sids:
                airport_point = self.current_airspace.navpoints.get(airport.sids[0])
            elif airport.stars:
                airport_point = self.current_airspace.navpoints.get(airport.stars[0])

            if airport_point:
                description = f"Airport: {airport_name}\nSIDs: {len(airport.sids)}, STARs: {len(airport.stars)}\nLocation: {airport_point.latitude}, {airport_point.longitude}"
                kml_content.append(_generate_kml_placemark_point(
                    airport_name, description, airport_point.longitude, airport_point.latitude, style_url="airportIcon"
                ))

        # Add Segments (Airways)
        for segment in self.current_airspace.navsegments:
            origin = self.current_airspace.navpoints.get(segment.origin_number)
            destination = self.current_airspace.navpoints.get(segment.destination_number)
            if origin and destination:
                description = f"Airway from {origin.name} to {destination.name}\nDistance: {segment.distance:.2f} km"
                coordinates = [(origin.longitude, origin.latitude), (destination.longitude, destination.latitude)]
                kml_content.append(_generate_kml_linestring(
                    f"Airway {origin.name}-{destination.name}", description, coordinates, style_url="airwayLine"
                ))

        kml_content.append(_generate_kml_footer())
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".kml",
            filetypes=[("KML files", "*.kml")],
            title=f"Save {self.current_airspace.name} Airspace KML File"
        )
        if file_path:
            try:
                # Use minidom for pretty printing
                xml_string = "\n".join(kml_content)
                dom = minidom.parseString(xml_string)
                pretty_xml_as_string = dom.toprettyxml(indent="  ")

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(pretty_xml_as_string)
                messagebox.showinfo("KML Export Success", f"Airspace KML saved to:\n{file_path}")
                self.open_kml_in_google_earth(file_path)
            except Exception as e:
                messagebox.showerror("KML Export Error", f"Failed to save KML file: {e}")

    def generate_kml_path(self):
        """Generates a KML file for the currently displayed shortest path."""
        if not self.current_display_path_nodes:
            messagebox.showwarning("No Path", "Please calculate and display a shortest path first to export to KML.")
            return

        kml_content = []
        kml_content.append(_generate_kml_header(f"Shortest Path: {self.current_display_path_nodes[0].name} to {self.current_display_path_nodes[-1].name}"))

        # Add Path Points
        for i, node in enumerate(self.current_display_path_nodes):
            name = f"Waypoint {i+1}: {node.name}"
            description = f"Lat: {node.latitude}, Lon: {node.longitude}"
            if i == 0:
                name = f"Start: {node.name}"
            elif i == len(self.current_display_path_nodes) - 1:
                name = f"End: {node.name}"
            kml_content.append(_generate_kml_placemark_point(
                name, description, node.longitude, node.latitude, style_url="navpointIcon"
            ))

        # Add Path LineString
        coordinates = [(node.longitude, node.latitude) for node in self.current_display_path_nodes]
        path_name = f"Path from {self.current_display_path_nodes[0].name} to {self.current_display_path_nodes[-1].name}"
        path_description = f"Total Distance: {self.current_display_path_cost:.2f} km"
        kml_content.append(_generate_kml_linestring(
            path_name, path_description, coordinates, style_url="pathLine"
        ))

        kml_content.append(_generate_kml_footer())

        file_path = filedialog.asksaveasfilename(
            defaultextension=".kml",
            filetypes=[("KML files", "*.kml")],
            title="Save Path KML File"
        )
        if file_path:
            try:
                # Use minidom for pretty printing
                xml_string = "\n".join(kml_content)
                dom = minidom.parseString(xml_string)
                pretty_xml_as_string = dom.toprettyxml(indent="  ")

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(pretty_xml_as_string)
                messagebox.showinfo("KML Export Success", f"Path KML saved to:\n{file_path}")
                self.open_kml_in_google_earth(file_path)
            except Exception as e:
                messagebox.showerror("KML Export Error", f"Failed to save KML file: {e}")

    def open_kml_in_google_earth(self, file_path):
        """Attempts to open a KML file with Google Earth (or default KML viewer)."""
        if os.path.exists(file_path):
            try:
                os.startfile(file_path) # For Windows
            except AttributeError:
                # For macOS/Linux, use subprocess.call
                import subprocess
                try:
                    subprocess.call(['open', file_path]) # macOS
                except FileNotFoundError:
                    try:
                        subprocess.call(['xdg-open', file_path]) # Linux (common)
                    except FileNotFoundError:
                        messagebox.showwarning("Open KML", "Could not automatically open KML file. Please open it manually with Google Earth.")
            except Exception as e:
                messagebox.showwarning("Open KML", f"Could not automatically open KML file: {e}\nPlease open it manually with Google Earth.")
        else:
            messagebox.showerror("File Not Found", f"KML file not found at:\n{file_path}")

# Part 5: Graph Editor Functions (Unchanged)

    # ===== Graph Editor Functions (Unchanged from your original script) =====
    def create_sample_graph(self):
        """Create the original example graph from interface2.py"""
        # Try to import from test_graphy.py first
        try:
            import sys
            import os
            
            # Add current directory to path to find test_graphy
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            from test_graphy import CreateGraph_1
            return CreateGraph_1()
        except ImportError:
            # If test_graphy.py is not available, create a basic example graph
            messagebox.showwarning("Missing File", 
                "test_graphy.py not found. Creating a basic example graph instead.\n"
                "Please ensure test_graphy.py with CreateGraph_1() function is in the same directory.")
            
            # Create a basic fallback graph
            graph = Graph()
            
            # Add nodes
            nodes_data = [
                ("A", 0, 0),
                ("B", 1, 1),
                ("C", 2, 0),
                ("D", 1, -1),
                ("E", 3, 1)
            ]
            
            for name, x, y in nodes_data:
                node = Node(name, x, y)
                graph.add_node(node)
            
            # Add segments
            segments_data = [
                ("AB", "A", "B"),
                ("AC", "A", "C"),
                ("BC", "B", "C"),
                ("BD", "B", "D"),
                ("CD", "C", "D"),
                ("CE", "C", "E")
            ]
            
            for seg_name, origin, dest in segments_data:
                AddSegment(graph, seg_name, origin, dest)
            
            return graph
    
    def load_example_graph(self):
        """Load the original example graph from test_graphy.py"""
        try:
            self.graph = self.create_sample_graph()
            self.refresh_graph()
            messagebox.showinfo("Success", "Example graph loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading example graph:\n{str(e)}")
    
    def load_custom_graph(self):
        """Load custom graph from test_graphy.py"""
        try:
            import sys
            import os
            
            # Add current directory to path to find test_graphy
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            from test_graphy import CreateGraph_2
            self.graph = CreateGraph_2()
            self.refresh_graph()
            messagebox.showinfo("Success", "Custom graph loaded successfully")
        except ImportError:
            messagebox.showerror("Error", 
                "test_graphy.py not found or CreateGraph_2() function not available.\n"
                "Please ensure test_graphy.py with CreateGraph_2() function is in the same directory.")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading custom graph:\n{str(e)}")
    
    def load_graph_from_file(self):
        """Load graph from a user-selected file"""
        file_path = filedialog.askopenfilename(
            title="Select graph file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialdir=os.getcwd()
        )
        
        if file_path:
            try:
                self.graph = ReadGraphFromFile(file_path)
                if self.graph:
                    self.refresh_graph()
                    messagebox.showinfo("Success", 
                        f"Graph loaded successfully from:\n{os.path.basename(file_path)}\n\n"
                        f"Nodes: {len(self.graph.nodes)}\n"
                        f"Segments: {len(self.graph.segments)}")
                else:
                    messagebox.showerror("Error", 
                        f"Could not load graph from file:\n{os.path.basename(file_path)}\n\n"
                        f"Please check that the file format is correct:\n"
                        f"NODES\n"
                        f"NodeName X Y\n"
                        f"...\n"
                        f"SEGMENTS\n"
                        f"SegmentName OriginNode DestNode Cost\n"
                        f"...")
            except Exception as e:
                messagebox.showerror("Error", 
                    f"Error loading graph from file:\n{os.path.basename(file_path)}\n\n"
                    f"Error details: {str(e)}\n\n"
                    f"Please check that the file format is correct.")
                print(f"Detailed error loading graph: {e}")
                import traceback
                traceback.print_exc()
    
    def save_graph(self):
        """Save graph to file"""
        if not self.graph:
            messagebox.showwarning("No Graph", "No graph to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save graph",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("NODES\n")
                    for node in self.graph.nodes.values():
                        f.write(f"{node.name} {node.x} {node.y}\n")
                    
                    f.write("\nSEGMENTS\n")
                    for segment in self.graph.segments.values():
                        f.write(f"{segment.name} {segment.origin.name} {segment.destination.name} {segment.cost}\n")
                
                messagebox.showinfo("Success", f"Graph saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving graph:\n{str(e)}")
    
    def refresh_graph(self):
        """Refresh the graph visualization with cool styling"""
        self.graph_figure.clear()
        ax = self.graph_figure.add_subplot(111)
        
        # Set cool background color for the plot
        ax.set_facecolor('#f0f8ff')  # Alice blue background
        
        if self.graph and hasattr(self.graph, 'nodes'):
            Plot(self.graph, ax)
            ax.set_title("Graph Visualization", fontsize=16, color=self.bg_primary, fontweight='bold', pad=20)
            ax.set_xlabel("X Coordinate", fontsize=12, color=self.text_dark)
            ax.set_ylabel("Y Coordinate", fontsize=12, color=self.text_dark)
            
            # Cool grid styling
            ax.grid(True, alpha=0.4, color=self.accent_cyan, linestyle='--', linewidth=0.5)
            ax.set_aspect('equal', adjustable='box')
            
            # Style the axes
            ax.spines['top'].set_color(self.accent_teal)
            ax.spines['bottom'].set_color(self.accent_teal)
            ax.spines['left'].set_color(self.accent_teal)
            ax.spines['right'].set_color(self.accent_teal)
            ax.tick_params(colors=self.text_dark)
            
        else:
            ax.text(0.5, 0.5, "No graph loaded", 
                   ha='center', va='center', fontsize=14, color=self.bg_secondary,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor=self.bg_light, alpha=0.8))
            ax.set_title("Graph Editor", fontsize=16, color=self.bg_primary, fontweight='bold')
            ax.axis('off')
        
        self.graph_canvas.draw()
        
        # Update info with cool styling
        if hasattr(self, 'graph_info_text'):
            self.graph_info_text.delete(1.0, tk.END)
            if self.graph:
                self.graph_info_text.insert(tk.END, f"📊 Nodes: {len(self.graph.nodes)}\n")
                self.graph_info_text.insert(tk.END, f"🔗 Segments: {len(self.graph.segments)}\n")
                self.graph_info_text.insert(tk.END, f"✅ Graph loaded and ready for analysis\n")
            else:
                self.graph_info_text.insert(tk.END, "ℹ️ No graph loaded\n")
    
    def show_graph_node(self):
        """Show a specific node and its neighbors"""
        if not self.graph:
            messagebox.showwarning("No Graph", "Please load a graph first")
            return
        
        node_name = self.search_node.get().strip()
        if not node_name:
            messagebox.showwarning("No Node", "Please enter a node name")
            return
        
        try:
            self.graph_figure.clear()
            ax = self.graph_figure.add_subplot(111)
            
            Plot(self.graph, ax)
            
            if not PlotNode(self.graph, node_name, ax):
                messagebox.showerror("Error", f"Node '{node_name}' not found")
                self.refresh_graph()
                return
            
            ax.set_title(f"Node: {node_name} and its neighbors")
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            
            self.graph_canvas.draw()
            
            # Update info
            node = self.graph.nodes[node_name]
            self.graph_info_text.delete(1.0, tk.END)
            self.graph_info_text.insert(tk.END, f"Selected Node: {node_name}\n")
            self.graph_info_text.insert(tk.END, f"Coordinates: ({node.x}, {node.y})\n")
            self.graph_info_text.insert(tk.END, f"Neighbors: {len(node.neighbors)}\n")
            for neighbor in node.neighbors:
                self.graph_info_text.insert(tk.END, f"  - {neighbor.name}\n")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error showing node:\n{str(e)}")
            self.refresh_graph()
    
    def add_node(self):
        """Add a new node to the graph"""
        if not self.graph:
            self.graph = Graph()
        
        node_name = simpledialog.askstring("Add Node", "Node name:")
        if not node_name:
            return
        
        if node_name in self.graph.nodes:
            messagebox.showerror("Error", f"Node '{node_name}' already exists")
            return
        
        try:
            x = float(simpledialog.askstring("Coordinates", "X coordinate:"))
            y = float(simpledialog.askstring("Coordinates", "Y coordinate:"))
            
            node = Node(node_name, x, y)
            AddNode(self.graph, node)
            self.refresh_graph()
            messagebox.showinfo("Success", f"Node '{node_name}' added successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Error adding node:\n{str(e)}")
    
    def add_segment(self):
        """Add a new segment to the graph"""
        if not self.graph:
            messagebox.showwarning("No Graph", "Please load or create a graph first")
            return
        
        origin = simpledialog.askstring("Add Segment", "Origin node:")
        if not origin or origin not in self.graph.nodes:
            messagebox.showerror("Error", f"Origin node '{origin}' not found")
            return
        
        destination = simpledialog.askstring("Add Segment", "Destination node:")
        if not destination or destination not in self.graph.nodes:
            messagebox.showerror("Error", f"Destination node '{destination}' not found")
            return
        
        try:
            cost_str = simpledialog.askstring("Add Segment", "Cost (leave empty for auto-calculation):")
            cost = None if not cost_str else float(cost_str)
            
            segment_name = f"{origin}-{destination}"
            # Check if segment already exists (bi-directional check for simplicity)
            exists = False
            for seg in self.graph.segments.values():
                if (seg.origin.name == origin and seg.destination.name == destination) or \
                   (seg.origin.name == destination and seg.destination.name == origin and cost is None): # If cost is auto-calculated, consider it a duplicate
                    exists = True
                    break
            
            if exists:
                messagebox.showwarning("Segment Exists", f"Segment from {origin} to {destination} already exists or its reverse with auto-calculated cost.")
                return

            AddSegment(self.graph, segment_name, origin, destination, cost)
            self.refresh_graph()
            messagebox.showinfo("Success", f"Segment '{segment_name}' added successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Error adding segment:\n{str(e)}")
    
    def delete_node(self):
        """Delete a node from the graph"""
        if not self.graph:
            messagebox.showwarning("No Graph", "Please load a graph first")
            return
        
        node_name = simpledialog.askstring("Delete Node", "Node name to delete:")
        if not node_name:
            return
        
        if node_name not in self.graph.nodes:
            messagebox.showerror("Error", f"Node '{node_name}' not found")
            return
        
        try:
            # Remove segments connected to this node
            segments_to_remove = [
                seg_name for seg_name, seg in self.graph.segments.items()
                if seg.origin.name == node_name or seg.destination.name == node_name
            ]
            
            for seg_name in segments_to_remove:
                del self.graph.segments[seg_name]
            
            # Remove from neighbor lists of other nodes
            node_to_delete = self.graph.nodes[node_name]
            for other_node in self.graph.nodes.values():
                # Create a new list for neighbors to modify it safely
                other_node.neighbors = [n for n in other_node.neighbors if n != node_to_delete]
            
            # Remove the node itself
            del self.graph.nodes[node_name]
            
            self.refresh_graph()
            messagebox.showinfo("Success", 
                f"Node '{node_name}' and {len(segments_to_remove)} connected segments deleted")
        except Exception as e:
            messagebox.showerror("Error", f"Error deleting node:\n{str(e)}")
    
    def find_closest_node(self):
        """Find the closest node to given coordinates"""
        if not self.graph:
            messagebox.showwarning("No Graph", "Please load a graph first")
            return
        
        try:
            x = float(simpledialog.askstring("Coordinates", "X coordinate:"))
            y = float(simpledialog.askstring("Coordinates", "Y coordinate:"))
            
            closest_node = GetClosest(self.graph, x, y)
            if closest_node:
                messagebox.showinfo("Closest Node", 
                    f"Closest node to ({x}, {y}):\n"
                    f"Name: {closest_node.name}\n"
                    f"Coordinates: ({closest_node.x}, {closest_node.y})\n"
                    f"Distance: {math.sqrt((closest_node.x - x)**2 + (closest_node.y - y)**2):.2f}")
                
                # Highlight the closest node
                self.graph_figure.clear()
                ax = self.graph_figure.add_subplot(111)
                Plot(self.graph, ax)
                
                # Mark the query point
                ax.plot(x, y, 'rx', markersize=15, markeredgewidth=3, label='Query Point')
                
                # Highlight closest node
                ax.plot(closest_node.x, closest_node.y, 'go', markersize=12, label='Closest Node')
                ax.plot([x, closest_node.x], [y, closest_node.y], 'g--', linewidth=2)
                
                ax.set_title(f"Closest Node to ({x}, {y})")
                ax.set_xlabel("X Coordinate")
                ax.set_ylabel("Y Coordinate")
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.set_aspect('equal', adjustable='box')
                
                self.graph_canvas.draw()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error finding closest node:\n{str(e)}")
    
    def show_graph_reachable(self):
        """Show reachable nodes from a starting node"""
        if not self.graph:
            messagebox.showwarning("No Graph", "Please load a graph first")
            return
        
        start_name = simpledialog.askstring("Reachable Nodes", "Starting node:")
        if not start_name or start_name not in self.graph.nodes:
            messagebox.showerror("Error", f"Starting node '{start_name}' not found")
            return
        
        try:
            reachable = FindReachableNodes(self.graph, start_name)
            
            self.graph_figure.clear()
            ax = self.graph_figure.add_subplot(111)
            
            Plot(self.graph, ax)
            
            # Highlight starting node
            start_node = self.graph.nodes[start_name]
            ax.plot(start_node.x, start_node.y, 'ro', markersize=12, label='Start Node')
            
            # Highlight reachable nodes
            for node in reachable:
                ax.plot(node.x, node.y, 'go', markersize=10, label='Reachable' if node == reachable[0] else '')
            
            ax.set_title(f"Nodes Reachable from {start_name}")
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_aspect('equal', adjustable='box')
            
            self.graph_canvas.draw()
            
            # Update info
            self.graph_info_text.delete(1.0, tk.END)
            self.graph_info_text.insert(tk.END, f"Starting Node: {start_name}\n")
            self.graph_info_text.insert(tk.END, f"Reachable Nodes: {len(reachable)}\n\n")
            for node in reachable:
                self.graph_info_text.insert(tk.END, f"- {node.name}\n")
            
            messagebox.showinfo("Result", 
                f"Found {len(reachable)} nodes reachable from {start_name}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error finding reachable nodes:\n{str(e)}")
    
    def show_graph_shortest_path(self):
        """Show shortest path between two nodes"""
        if not self.graph:
            messagebox.showwarning("No Graph", "Please load a graph first")
            return
        
        start_name = simpledialog.askstring("Shortest Path", "Starting node:")
        if not start_name or start_name not in self.graph.nodes:
            messagebox.showerror("Error", f"Starting node '{start_name}' not found")
            return
        
        end_name = simpledialog.askstring("Shortest Path", "Destination node:")
        if not end_name or end_name not in self.graph.nodes:
            messagebox.showerror("Error", f"Destination node '{end_name}' not found")
            return
        
        try:
            path = FindShortestPath(self.graph, start_name, end_name)
            
            if not path:
                messagebox.showinfo("No Path", f"No path found from {start_name} to {end_name}")
                return
            
            self.graph_figure.clear()
            ax = self.graph_figure.add_subplot(111)
            
            Plot(self.graph, ax)
            PlotPath(self.graph, path, ax)
            
            ax.set_title(f"Shortest Path: {start_name} → {end_name}")
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            
            self.graph_canvas.draw()
            
            # Update info
            self.graph_info_text.delete(1.0, tk.END)
            self.graph_info_text.insert(tk.END, f"Path: {start_name} → {end_name}\n")
            self.graph_info_text.insert(tk.END, f"Total Cost: {path.cost:.2f}\n")
            self.graph_info_text.insert(tk.END, f"Nodes in Path: {len(path.nodes)}\n\n")
            
            path_str = " → ".join([node.name for node in path.nodes])
            self.graph_info_text.insert(tk.END, f"Route: {path_str}\n")
            
            messagebox.showinfo("Shortest Path", 
                f"Path found!\n"
                f"Route: {path_str}\n"
                f"Total cost: {path.cost:.2f}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error finding shortest path:\n{str(e)}")

# Part 6: Advanced Features (Simulation and Weather) - Updated for KML Export

    # ===== Advanced Features (Existing methods with enhancements) =====

    def spawn_aircraft(self):
        """Spawns a new simulated aircraft with a random origin and destination."""
        if not self.current_airspace or not self.current_airspace.navpoints:
            messagebox.showwarning("No Airspace", "Load an airspace first to spawn aircraft.")
            return

        nav_points_list = list(self.current_airspace.navpoints.values())
        if len(nav_points_list) < 2:
            messagebox.showwarning("Not Enough Points", "Need at least two navigation points to spawn an aircraft.")
            return

        # Select random origin and destination that are not the same
        origin_point = random.choice(nav_points_list)
        destination_point = random.choice(nav_points_list)
        while origin_point == destination_point:
            destination_point = random.choice(nav_points_list)

        aircraft_id = f"AC{len(self.simulated_aircrafts) + 1:03d}" # Format ID with leading zeros
        new_aircraft = SimulatedAircraft(aircraft_id, origin_point, destination_point)
        
        if new_aircraft.calculate_path(self.current_airspace, destination_point):
            self.simulated_aircrafts.append(new_aircraft)
            if hasattr(self, 'advanced_info_text'):
                self.advanced_info_text.insert(tk.END, f"✈️ Spawned Aircraft {aircraft_id}: {origin_point.name} -> {destination_point.name}\n")
            self.redraw_animated_elements() # Initial plot of the new aircraft
        else:
            if hasattr(self, 'advanced_info_text'):
                self.advanced_info_text.insert(tk.END, f"❌ Failed to spawn Aircraft {aircraft_id}: No path found.\n")

    def start_traffic_simulation(self):
        """Starts the animated air traffic simulation."""
        if self.simulation_active:
            messagebox.showinfo("Simulation Status", "Simulation is already running.")
            return
        if not self.current_airspace:
            messagebox.showwarning("No Airspace", "Load an airspace first to start simulation.")
            return
        if not self.simulated_aircrafts and not messagebox.askyesno("No Aircraft", "No aircraft spawned. Start simulation anyway?"):
            return # Don't start if no aircraft and user declines

        self.simulation_active = True
        self.last_update_time = time.time() # Initialize time
        self.simulation_start_epoch = time.time() # Capture start time for KML gx:Track
        if hasattr(self, 'advanced_info_text'):
            self.advanced_info_text.insert(tk.END, "🚀 Starting air traffic simulation...\n")
        
        # Ensure plot is ready (especially if coming from a different tab)
        self.show_complete_airspace() # Re-plot base airspace (removes old dynamic artists if any)
        self.redraw_animated_elements() # Initial draw of aircraft/weather on base map
        
        self.update_traffic_simulation() # Start the recursive loop

    def stop_traffic_simulation(self):
        """Stops the animated air traffic simulation."""
        if not self.simulation_active:
            return # Don't show info if just switching tabs etc.

        self.simulation_active = False
        if self.animation_job:
            self.root.after_cancel(self.animation_job)
            self.animation_job = None
        if hasattr(self, 'advanced_info_text'):
            self.advanced_info_text.insert(tk.END, "🛑 Stopping air traffic simulation.\n")
        self.clear_conflict_warnings() # Clear any warning bubbles
        # If stopping, remove all dynamic elements from the plot
        self.clear_aircraft_plot_artists()
        self.clear_weather_plot_artists()
        self.airspace_canvas.draw_idle() # Redraw the static airspace without animated elements

    def clear_aircraft_simulation(self):
        """Clears all simulated aircraft and their plots."""
        self.stop_traffic_simulation() # Ensure simulation is stopped first
        self.clear_aircraft_plot_artists() # Remove artists from plot
        self.simulated_aircrafts = [] # Clear the list of aircraft objects
        self.clear_conflict_warnings() # Clear any remaining conflict warnings
        self.airspace_canvas.draw_idle() # Redraw to show changes
        if hasattr(self, 'advanced_info_text'):
            self.advanced_info_text.insert(tk.END, "All aircraft simulation data and visuals cleared.\n")

    def clear_aircraft_plot_artists(self):
        """Helper to remove all aircraft related matplotlib artists."""
        for ac_id, artists in self.aircraft_plot_artists.items():
            if artists.get('icon'): artists['icon'].remove()
            if artists.get('text'): artists['text'].remove()
            if artists.get('trail'): artists['trail'].remove()
            if artists.get('path_line'): artists['path_line'].remove() # Remove path line if it was drawn
        self.aircraft_plot_artists.clear()

    def update_simulation_speed(self, val):
        """Updates the simulation speed based on the slider value."""
        # The scale's `command` passes the value as a string. Convert it.
        new_speed = float(val)
        if new_speed > 0:
            self.simulation_speed.set(new_speed)

    def redraw_animated_elements(self):
        """Redraws all dynamic elements efficiently using Matplotlib artists."""
        if not hasattr(self, 'airspace_ax') or not self.airspace_ax:
            return

        # Update or create artists for aircraft
        # Iterate over a copy of keys to safely remove artists if aircraft become inactive
        current_aircraft_ids = {ac.aircraft_id for ac in self.simulated_aircrafts}
        for ac_id in list(self.aircraft_plot_artists.keys()): # Iterate over keys to allow removal
            if ac_id not in current_aircraft_ids:
                artists = self.aircraft_plot_artists.pop(ac_id)
                if artists.get('icon'): artists['icon'].remove()
                if artists.get('text'): artists['text'].remove()
                if artists.get('trail'): artists['trail'].remove()
                if artists.get('path_line'): artists['path_line'].remove()
        
        for aircraft in self.simulated_aircrafts:
            lon, lat = aircraft.get_current_coordinates()
            if lon is not None and lat is not None:
                # Update existing artists or create new ones
                if aircraft.aircraft_id not in self.aircraft_plot_artists:
                    # Initial creation of artists
                    marker_style = 'P' 
                    color = 'blue'
                    if isinstance(aircraft, EnhancedAircraft):
                        color = aircraft.color
                        if aircraft.firejet_warning_active:
                            color = 'red'
                            marker_style = '▲'  # Warning triangle
                    
                    icon_artist, = self.airspace_ax.plot(lon, lat, marker_style, markersize=10, 
                                                            color=color, alpha=0.8, zorder=4)
                    text_artist = self.airspace_ax.text(lon + 0.01, lat + 0.01, aircraft.aircraft_id,
                                                        fontsize=7, color='navy', zorder=5)
                    trail_artist, = self.airspace_ax.plot([], [], color='cyan', linewidth=1.5, alpha=0.6, zorder=3)
                    # The full path line (current path or rerouted path)
                    path_line_artist, = self.airspace_ax.plot([], [], linestyle=':', color='gray', linewidth=0.5, alpha=0.5, zorder=1) 
                    
                    self.aircraft_plot_artists[aircraft.aircraft_id] = {
                        'icon': icon_artist, 'text': text_artist, 'trail': trail_artist, 'path_line': path_line_artist
                    }
                
                artists = self.aircraft_plot_artists[aircraft.aircraft_id]

                # Update icon position and style
                artists['icon'].set_xdata(lon)
                artists['icon'].set_ydata(lat)
                
                # Enhanced aircraft display
                if isinstance(aircraft, EnhancedAircraft):
                    if aircraft.firejet_warning_active:
                        artists['icon'].set_color('red')
                        artists['icon'].set_marker('X')  # Warning marker
                        artists['icon'].set_markersize(15)
                    else:
                        artists['icon'].set_color(aircraft.color)
                        artists['icon'].set_markersize(10 * aircraft.current_scale)
                else:
                    artists['icon'].set_color('blue' if not aircraft.rerouting else 'orange')
                
                # Change marker based on direction (simplified)
                if aircraft.next_navpoint and not (isinstance(aircraft, EnhancedAircraft) and aircraft.firejet_warning_active):
                    delta_lon = aircraft.next_navpoint.longitude - aircraft.current_navpoint.longitude
                    delta_lat = aircraft.next_navpoint.latitude - aircraft.current_navpoint.latitude
                    if abs(delta_lon) > abs(delta_lat):
                        artists['icon'].set_marker('>' if delta_lon > 0 else '<')
                    else:
                        artists['icon'].set_marker('^' if delta_lat > 0 else 'v')
                elif not aircraft.next_navpoint: # At destination
                    artists['icon'].set_marker('X') # Mark arrived aircraft
                    artists['icon'].set_color('green')

                # Update text position and color
                text_color = 'navy'
                if isinstance(aircraft, EnhancedAircraft) and aircraft.firejet_warning_active:
                    text_color = 'red'
                elif aircraft.rerouting:
                    text_color = 'darkorange'
                    
                artists['text'].set_position((lon + 0.01, lat + 0.01))
                artists['text'].set_color(text_color)

                # Update trail
                if aircraft.current_trail_coords:
                    trail_lons, trail_lats = zip(*aircraft.current_trail_coords)
                    artists['trail'].set_xdata(trail_lons)
                    artists['trail'].set_ydata(trail_lats)
                    
                    trail_color = 'cyan'
                    if isinstance(aircraft, EnhancedAircraft) and aircraft.firejet_warning_active:
                        trail_color = 'red'
                    elif aircraft.rerouting:
                        trail_color = 'yellow'
                        
                    artists['trail'].set_color(trail_color)
                else:
                    artists['trail'].set_xdata([])
                    artists['trail'].set_ydata([])

                # Update current/original path line
                if aircraft.path_to_destination:
                    path_lon_coords = [p.longitude for p in aircraft.path_to_destination]
                    path_lat_coords = [p.latitude for p in aircraft.path_to_destination]
                    artists['path_line'].set_xdata(path_lon_coords)
                    artists['path_line'].set_ydata(path_lat_coords)
                    artists['path_line'].set_color('gray' if not aircraft.rerouting else 'orange')
                    artists['path_line'].set_linestyle(':' if not aircraft.rerouting else '--')

        # Update or create artists for weather zones
        current_weather_names = {zone.name for zone in self.weather_zones}
        for name in list(self.weather_plot_artists.keys()):
            if name not in current_weather_names:
                artist = self.weather_plot_artists.pop(name)
                if artist: artist.remove()

        for zone in self.weather_zones:
            if zone.name not in self.weather_plot_artists:
                # Create new patch for polygon
                polygon_coords = zone.get_mpl_polygon_coords()
                color_map = {'light': 'lightblue', 'moderate': 'orange', 'severe': 'red'}
                alpha_map = {'light': 0.3, 'moderate': 0.5, 'severe': 0.7}
                poly_artist = self.airspace_ax.add_patch(
                    plt.Polygon(polygon_coords, closed=True, fc=color_map[zone.severity], 
                                ec=color_map[zone.severity], alpha=alpha_map[zone.severity], zorder=2)
                )
                self.weather_plot_artists[zone.name] = poly_artist
            else:
                # Update existing patch
                self.weather_plot_artists[zone.name].set_xy(zone.get_mpl_polygon_coords())
                
        # Canvas redraw (optimized)
        self.airspace_canvas.draw_idle()

    def detect_conflicts(self):
        """Predicts and visualizes potential mid-air conflicts."""
        self.clear_conflict_warnings() # Clear previous warnings to redraw fresh ones

        conflict_threshold_km = 10 # Aircraft within 10 km
        time_look_ahead_hours = 0.05 # Look 3 minutes ahead (0.05 hours) for short-term prediction

        conflicts_found = []

        # Iterate over all pairs of active aircraft
        active_aircrafts = [ac for ac in self.simulated_aircrafts if ac.is_active]
        for i, ac1 in enumerate(active_aircrafts):
            for j, ac2 in enumerate(active_aircrafts):
                if i >= j: continue # Avoid self-comparison and duplicate pairs

                lon1_fut, lat1_fut = ac1.get_future_coordinates(time_look_ahead_hours)
                lon2_fut, lat2_fut = ac2.get_future_coordinates(time_look_ahead_hours)

                if lon1_fut is None or lon2_fut is None: continue

                # Calculate distance between future positions using Haversine
                R = 6371 # Radius of Earth in kilometers
                lat1_rad = math.radians(lat1_fut)
                lon1_rad = math.radians(lon1_fut)
                lat2_rad = math.radians(lat2_fut)
                lon2_rad = math.radians(lon2_fut)

                dlon = lon2_rad - lon1_rad
                dlat = lat2_rad - lat1_rad

                a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                predicted_dist_km = R * c

                # Simple altitude check (if within 1000ft / ~0.3km)
                altitude_difference_meters = abs(ac1.altitude - ac2.altitude)
                if altitude_difference_meters > 300: # If altitude difference is more than 300 meters (~1000 ft)
                    continue # Not a conflict due to vertical separation

                if predicted_dist_km < conflict_threshold_km:
                    conflicts_found.append((ac1, ac2, (lon1_fut, lat1_fut), (lon2_fut, lat2_fut)))

        # Reset conflict_warning for aircraft that are no longer in conflicts
        for ac in self.simulated_aircrafts:
            ac.conflict_warning = False 
        
        # Now mark aircraft that ARE in conflict
        for ac1, ac2, _, _ in conflicts_found:
            ac1.conflict_warning = True 
            ac2.conflict_warning = True

        # Plot conflict warnings
        ax = self.airspace_ax
        for ac1, ac2, (lon1_fut, lat1_fut), (lon2_fut, lat2_fut) in conflicts_found:
            center_lon = (lon1_fut + lon2_fut) / 2
            center_lat = (lat1_fut + lat2_fut) / 2
            
            # Draw a circle (approximate radius in degrees)
            radius_degrees = conflict_threshold_km / 111.0 # Convert km to degrees
            circle = plt.Circle((center_lon, center_lat), radius_degrees, 
                                color='red', alpha=0.3 + (math.sin(time.time()*5)*0.1), # Slight pulsating alpha effect
                                ec='red', lw=2, linestyle='--', zorder=6) # Red outline
            ax.add_patch(circle)
            self.conflict_plot_artists.append(circle)

            # Draw connecting line
            line, = ax.plot([lon1_fut, lon2_fut], [lat1_fut, lat2_fut], 
                            color='red', linestyle='--', linewidth=2, zorder=7)
            self.conflict_plot_artists.append(line)
            
            # Update info text (only print new alerts to avoid spam)
            if hasattr(self, 'advanced_info_text'):
                alert_message = f"🚨 CONFLICT ALERT: {ac1.aircraft_id} (FL{ac1.altitude/100:.0f}) & {ac2.aircraft_id} (FL{ac2.altitude/100:.0f}) predicted at ({center_lon:.2f}, {center_lat:.2f})\n"
                if alert_message not in self.advanced_info_text.get(1.0, tk.END): # Basic check to prevent duplicate messages
                    self.advanced_info_text.insert(tk.END, alert_message)

        if conflicts_found:
            self.airspace_canvas.draw_idle()

    def clear_conflict_warnings(self):
        """Removes all conflict warning visualizations from the plot."""
        for artist in self.conflict_plot_artists:
            if hasattr(self, 'airspace_ax') and self.airspace_ax:
                try:
                    artist.remove()
                except:
                    pass  # Artist might already be removed
        self.conflict_plot_artists = [] # Clear the list of artists

    def generate_weather_zones(self):
        """Generates and plots random, animated weather zones."""
        if not self.current_airspace:
            messagebox.showwarning("No Airspace", "Load an airspace first to generate weather.")
            return

        self.clear_weather_zones() # Clear existing weather plots and data
        
        # Get bounds of current airspace for random weather generation
        lats = [np.latitude for np in self.current_airspace.navpoints.values()]
        lons = [np.longitude for np in self.current_airspace.navpoints.values()]

        if not lats or not lons:
            messagebox.showwarning("No Data", "Airspace data not sufficient to generate weather.")
            return

        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        # Extend bounds slightly for weather generation so they can move into view
        lon_range = max_lon - min_lon
        lat_range = max_lat - min_lat
        min_lon -= lon_range * 0.2
        max_lon += lon_range * 0.2
        min_lat -= lat_range * 0.2
        max_lat += lat_range * 0.2

        num_zones = random.randint(1, 3) # Generate 1 to 3 weather zones
        for i in range(num_zones):
            center_lon = random.uniform(min_lon, max_lon)
            center_lat = random.uniform(min_lat, max_lat)
            size_deg = random.uniform(0.5, 2.0) # Size in degrees
            severity = random.choice(["light", "moderate", "severe"])
            speed = random.uniform(10, 50) # kph
            direction = random.uniform(0, 360) # degrees

            # Create simple square/rectangular weather zones for demonstration
            # Vertices ordered to form a polygon: (bottom-left, bottom-right, top-right, top-left, close)
            vertices = [
                (center_lon - size_deg/2, center_lat - size_deg/2),
                (center_lon + size_deg/2, center_lat - size_deg/2),
                (center_lon + size_deg/2, center_lat + size_deg/2),
                (center_lon - size_deg/2, center_lat + size_deg/2),
                (center_lon - size_deg/2, center_lat - size_deg/2) # Close the polygon
            ]
            zone_name = f"WeatherZone_{i+1}"
            new_zone = WeatherZone(zone_name, vertices, severity, speed, direction)
            self.weather_zones.append(new_zone)
            if hasattr(self, 'advanced_info_text'):
                self.advanced_info_text.insert(tk.END, f"☁️ Generated {severity} weather zone '{zone_name}' moving at {speed:.1f} kph.\n")
        
        self.redraw_animated_elements() # Plot immediately

    def clear_weather_zones(self):
        """Clears all generated weather zones and their plots."""
        self.clear_weather_plot_artists() # Remove artists from plot
        self.weather_zones = [] # Clear the list of weather objects
        self.airspace_canvas.draw_idle() # Redraw to show changes
        if hasattr(self, 'advanced_info_text'):
            self.advanced_info_text.insert(tk.END, "Weather zones cleared.\n")

    def clear_weather_plot_artists(self):
        """Helper to remove all weather related matplotlib artists."""
        for name, artist in self.weather_plot_artists.items():
            if artist: 
                try:
                    artist.remove()
                except:
                    pass  # Artist might already be removed
        self.weather_plot_artists.clear()

    def dynamic_reroute_for_weather(self):
        """Checks for aircraft intersecting severe weather zones and attempts to reroute them."""
        if not self.current_airspace: return

        for aircraft in self.simulated_aircrafts:
            if not aircraft.is_active: continue

            current_lon, current_lat = aircraft.get_current_coordinates()
            if current_lon is None: continue

            # Determine if aircraft is in a severe weather zone or will enter one soon
            is_in_severe_weather = False
            for zone in self.weather_zones:
                if zone.severity == "severe":
                    # Check current position
                    if zone.contains_point(current_lon, current_lat):
                        is_in_severe_weather = True
                        break
                    # Check next waypoint (simple look-ahead)
                    if aircraft.next_navpoint:
                        if zone.contains_point(aircraft.next_navpoint.longitude, aircraft.next_navpoint.latitude):
                            is_in_severe_weather = True
                            break

            # If rerouting is needed and aircraft is not already rerouting
            if is_in_severe_weather and not aircraft.rerouting:
                aircraft.rerouting = True
                if hasattr(self, 'advanced_info_text'):
                    self.advanced_info_text.insert(tk.END, f"⚠️ Aircraft {aircraft.aircraft_id} detecting severe weather. Rerouting...\n")

                excluded_points = set()
                excluded_segments = set()
                
                # Identify points and segments to exclude for rerouting
                for zone in self.weather_zones:
                    if zone.severity == "severe":
                        # Exclude all navpoints within the severe weather zone
                        for np_id, nav_point in self.current_airspace.navpoints.items():
                            if zone.contains_point(nav_point.longitude, nav_point.latitude):
                                excluded_points.add(np_id)
                        
                        # Exclude segments that pass through the severe weather zone
                        # Simplified: checking midpoint of segment
                        for segment in self.current_airspace.navsegments:
                            origin_np = self.current_airspace.navpoints.get(segment.origin_number)
                            dest_np = self.current_airspace.navpoints.get(segment.destination_number)
                            if origin_np and dest_np:
                                mid_lon = (origin_np.longitude + dest_np.longitude) / 2
                                mid_lat = (origin_np.latitude + dest_np.latitude) / 2
                                if zone.contains_point(mid_lon, mid_lat):
                                    excluded_segments.add((origin_np.number, dest_np.number))

                # Attempt to calculate a new path avoiding the identified hazards
                # Start from the current actual navpoint of the aircraft
                if aircraft.calculate_path(self.current_airspace, aircraft.path_to_destination[-1], 
                                          excluded_points=excluded_points, excluded_segments=excluded_segments):
                    if hasattr(self, 'advanced_info_text'):
                        self.advanced_info_text.insert(tk.END, f"✅ Aircraft {aircraft.aircraft_id} rerouted successfully.\n")
                else:
                    if hasattr(self, 'advanced_info_text'):
                        self.advanced_info_text.insert(tk.END, f"❌ Aircraft {aircraft.aircraft_id} could not be rerouted. Considering emergency measures.\n")
                    # For a real application, you might raise an alarm or perform an emergency landing logic
                    aircraft.is_active = False # Deactivate if no reroute possible (simplified for demo)
            
            # If aircraft was rerouting but is now clear of severe weather (and not in a new severe zone)
            elif aircraft.rerouting and not is_in_severe_weather:
                if hasattr(self, 'advanced_info_text'):
                    self.advanced_info_text.insert(tk.END, f"🟢 Aircraft {aircraft.aircraft_id} now clear of severe weather. Resuming optimal path.\n")
                aircraft.rerouting = False
                # Recalculate original optimal path from current position to destination
                aircraft.calculate_path(self.current_airspace, aircraft.path_to_destination[-1])

    def generate_kml_enhanced_aircraft_models(self):
        """Generate KML file with 3D aircraft models"""
        if not self.simulated_aircrafts:
            messagebox.showwarning("No Aircraft", "No simulated aircraft to export.")
            return
            
        kml_content = []
        kml_content.append(_generate_kml_header("Enhanced Aircraft 3D Models"))
        
        # Add 3D models for each aircraft
        for aircraft in self.simulated_aircrafts:
            if isinstance(aircraft, EnhancedAircraft):
                kml_content.append(aircraft.get_kml_model_3d())
        
        kml_content.append(_generate_kml_footer())
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".kml",
            filetypes=[("KML files", "*.kml")],
            title="Save Enhanced Aircraft Models KML"
        )
        if file_path:
            try:
                xml_string = "\n".join(kml_content)
                dom = minidom.parseString(xml_string)
                pretty_xml_as_string = dom.toprettyxml(indent="  ")

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(pretty_xml_as_string)
                messagebox.showinfo("KML Export Success", f"Enhanced Aircraft Models KML saved to:\n{file_path}")
                self.open_kml_in_google_earth(file_path)
            except Exception as e:
                messagebox.showerror("KML Export Error", f"Failed to save KML file: {e}")

    def simulate_interception_scenario(self):
        """
        Simulates a complete interception scenario:
        1. Civilian aircraft crosses restricted airspace
        2. Fighter jets are scrambled for interception
        3. Full scenario exported to KML with gx:Track
        """
        if not self.current_airspace or not self.current_airspace.navpoints:
            messagebox.showwarning("No Airspace", "Load an airspace first to run interception scenario.")
            return
            
        # Initialize 3D zones if not already done
        if not hasattr(self, 'airspace_zones_3d') or not self.airspace_zones_3d:
            self.setup_3d_airspace_zones()
        
        # Find restricted zone for the scenario
        restricted_zone = None
        for zone in self.airspace_zones_3d:
            if zone.zone_type == 'RESTRICTED':
                restricted_zone = zone
                break
        
        if not restricted_zone:
            messagebox.showwarning("No Restricted Zone", "No restricted airspace zone found. Setting up zones first.")
            self.setup_3d_airspace_zones()
            for zone in self.airspace_zones_3d:
                if zone.zone_type == 'RESTRICTED':
                    restricted_zone = zone
                    break
        
        # Clear existing simulation
        self.clear_aircraft_simulation()
        
        # Setup scenario participants
        self.interception_scenario_data = {
            'civilian_aircraft': None,
            'interceptor_jets': [],
            'scenario_start_time': time.time(),
            'interception_triggered': False,
            'interception_time': None,
            'escort_phase': False,
            'scenario_complete': False
        }
        
        # Create civilian aircraft that will violate restricted airspace
        nav_points_list = list(self.current_airspace.navpoints.values())
        
        # ENHANCED PATH FINDING: Try multiple strategies
        civilian_aircraft = None
        attempts = 0
        max_attempts = 10
        
        while civilian_aircraft is None and attempts < max_attempts:
            attempts += 1
            
            if attempts <= 3:
                # Strategy 1: Try to find points on opposite sides of restricted zone
                zone_center_lon = sum(lon for lon, lat in restricted_zone.vertices) / len(restricted_zone.vertices)
                zone_center_lat = sum(lat for lon, lat in restricted_zone.vertices) / len(restricted_zone.vertices)
                
                origin_candidates = []
                dest_candidates = []
                
                for np in nav_points_list:
                    if not restricted_zone._point_in_polygon(np.longitude, np.latitude):
                        dist_to_center = math.sqrt((np.longitude - zone_center_lon)**2 + (np.latitude - zone_center_lat)**2)
                        if dist_to_center > 0.2:  # Reduced margin
                            if np.longitude < zone_center_lon:
                                origin_candidates.append(np)
                            else:
                                dest_candidates.append(np)
                
                if origin_candidates and dest_candidates:
                    origin_point = random.choice(origin_candidates)
                    dest_point = random.choice(dest_candidates)
                else:
                    continue  # Try next strategy
                    
            elif attempts <= 6:
                # Strategy 2: Find any two distant points
                sorted_points = list(nav_points_list)
                random.shuffle(sorted_points)
                
                for i, origin_point in enumerate(sorted_points[:len(sorted_points)//2]):
                    for dest_point in sorted_points[len(sorted_points)//2:]:
                        # Check if they're sufficiently far apart
                        dist = math.sqrt((origin_point.longitude - dest_point.longitude)**2 + 
                                       (origin_point.latitude - dest_point.latitude)**2)
                        if dist > 0.5:  # Minimum distance
                            break
                    if dist > 0.5:
                        break
            else:
                # Strategy 3: Just use any two different points
                origin_point = random.choice(nav_points_list)
                dest_point = random.choice(nav_points_list)
                while origin_point == dest_point:
                    dest_point = random.choice(nav_points_list)
            
            # Try to create aircraft with these points
            test_aircraft = EnhancedAircraft(
                "CVL001", origin_point, dest_point, "AIRLINER"
            )
            test_aircraft.speed_kph = 850
            test_aircraft.altitude = 35000 * 0.3048
            
            # Test if path calculation works
            if test_aircraft.calculate_path(self.current_airspace, dest_point):
                civilian_aircraft = test_aircraft
                if hasattr(self, 'advanced_info_text'):
                    self.advanced_info_text.insert(tk.END, 
                        f"✅ Path found on attempt {attempts}: {origin_point.name} → {dest_point.name}\n")
            else:
                if hasattr(self, 'advanced_info_text'):
                    self.advanced_info_text.insert(tk.END, 
                        f"❌ Attempt {attempts} failed: {origin_point.name} → {dest_point.name}\n")
        
        if civilian_aircraft is None:
            # FALLBACK: Create a simple direct-route aircraft without pathfinding
            if hasattr(self, 'advanced_info_text'):
                self.advanced_info_text.insert(tk.END, 
                    "⚠️ Using fallback: Creating direct route without pathfinding\n")
            
            origin_point = random.choice(nav_points_list)
            dest_point = random.choice(nav_points_list)
            while origin_point == dest_point:
                dest_point = random.choice(nav_points_list)
            
            civilian_aircraft = EnhancedAircraft(
                "CVL001", origin_point, dest_point, "AIRLINER"
            )
            civilian_aircraft.speed_kph = 850
            civilian_aircraft.altitude = 35000 * 0.3048
            
            # Manually create a simple 2-point path
            civilian_aircraft.path_to_destination = [origin_point, dest_point]
            civilian_aircraft.current_navpoint = origin_point
            civilian_aircraft.next_navpoint = dest_point
            civilian_aircraft.current_segment_progress = 0.0
            civilian_aircraft.is_active = True
            
            if hasattr(self, 'advanced_info_text'):
                self.advanced_info_text.insert(tk.END, 
                    f"✅ Fallback route created: {origin_point.name} → {dest_point.name}\n")
        
        # Add aircraft to simulation
        self.simulated_aircrafts.append(civilian_aircraft)
        self.interception_scenario_data['civilian_aircraft'] = civilian_aircraft
        
        if hasattr(self, 'advanced_info_text'):
            self.advanced_info_text.insert(tk.END, 
                "🎯 INTERCEPTION SCENARIO INITIATED\n")
            self.advanced_info_text.insert(tk.END, 
                f"📡 Civilian aircraft CVL001 on route {civilian_aircraft.start_navpoint.name} → {civilian_aircraft.end_navpoint.name}\n")
            self.advanced_info_text.insert(tk.END, 
                f"🛡️ Monitoring for restricted airspace violations...\n")
        
        # Start the enhanced interception simulation
        self.start_interception_simulation()

    def start_interception_simulation(self):
        """Start the specialized interception simulation with enhanced monitoring"""
        if self.simulation_active:
            messagebox.showinfo("Simulation Status", "Stopping current simulation to start interception scenario.")
            self.stop_traffic_simulation()
        
        self.simulation_active = True
        self.last_update_time = time.time()
        self.interception_scenario_data['scenario_start_time'] = time.time()
        
        # Ensure plot is ready
        self.show_complete_airspace()
        self.redraw_animated_elements()
        
        if hasattr(self, 'advanced_info_text'):
            self.advanced_info_text.insert(tk.END, "🚀 INTERCEPTION SIMULATION ACTIVE\n")
        
        # Start the enhanced update loop
        self.update_interception_simulation()

    def update_interception_simulation(self):
        """Enhanced simulation update specifically for interception scenarios"""
        if not self.simulation_active or not hasattr(self, 'interception_scenario_data'):
            return

        current_time_epoch = time.time()
        delta_time_seconds = (current_time_epoch - self.last_update_time) * self.simulation_speed.get()
        delta_time_hours = delta_time_seconds / 3600.0
        self.last_update_time = current_time_epoch

        # Update aircraft positions
        for aircraft in self.simulated_aircrafts:
            if aircraft.is_active:
                aircraft.update_position(delta_time_hours, current_time_epoch)

        # Check for restricted airspace violation
        self.check_interception_triggers()
        
        # Update interception logic
        self.update_interception_logic(delta_time_hours, current_time_epoch)
        
        # Standard conflict detection and weather rerouting
        self.monitor_3d_airspace_violations()
        self.detect_conflicts()
        
        # Redraw all elements
        self.redraw_animated_elements()

        # Continue simulation
        self.animation_job = self.root.after(self.animation_interval_ms, self.update_interception_simulation)

    def check_interception_triggers(self):
        """Check if civilian aircraft has triggered interception protocols"""
        if not hasattr(self, 'interception_scenario_data') or self.interception_scenario_data['interception_triggered']:
            return
        
        civilian_aircraft = self.interception_scenario_data['civilian_aircraft']
        if not civilian_aircraft or not civilian_aircraft.is_active:
            return
        
        # Check if aircraft is in restricted zone
        for zone in self.airspace_zones_3d:
            if zone.zone_type == 'RESTRICTED' and zone.contains_aircraft(civilian_aircraft):
                self.trigger_interception(zone)
                break

    def trigger_interception(self, restricted_zone):
        """Trigger the interception sequence when aircraft violates restricted airspace"""
        self.interception_scenario_data['interception_triggered'] = True
        self.interception_scenario_data['interception_time'] = time.time()
        
        civilian_aircraft = self.interception_scenario_data['civilian_aircraft']
        
        if hasattr(self, 'advanced_info_text'):
            self.advanced_info_text.insert(tk.END, 
                "🚨 RESTRICTED AIRSPACE VIOLATION DETECTED!\n")
            self.advanced_info_text.insert(tk.END, 
                f"📍 Aircraft CVL001 has entered {restricted_zone.name}\n")
            self.advanced_info_text.insert(tk.END, 
                "🛩️ SCRAMBLING INTERCEPTOR JETS...\n")
        
        # Create interceptor jets
        self.create_interceptor_jets(civilian_aircraft, restricted_zone)
        
        # Activate firejet warning for civilian aircraft
        civilian_aircraft.activate_firejet_warning()

    def create_interceptor_jets(self, civilian_aircraft, restricted_zone):
        """Create and deploy interceptor fighter jets"""
        # Find nearby navpoints to launch interceptors from
        nav_points_list = list(self.current_airspace.navpoints.values())
        civilian_lon, civilian_lat = civilian_aircraft.get_current_coordinates()
        
        if civilian_lon is None or civilian_lat is None:
            # Use aircraft's current navpoint if coordinates not available
            civilian_lon = civilian_aircraft.current_navpoint.longitude
            civilian_lat = civilian_aircraft.current_navpoint.latitude
        
        # Sort navpoints by distance to civilian aircraft
        nearby_points = []
        for np in nav_points_list:
            dist = math.sqrt((np.longitude - civilian_lon)**2 + (np.latitude - civilian_lat)**2)
            if not restricted_zone._point_in_polygon(np.longitude, np.latitude):  # Not in restricted zone
                nearby_points.append((dist, np))
        
        nearby_points.sort(key=lambda x: x[0])
        
        # Create up to 2 interceptor jets from different positions
        interceptor_count = min(2, len(nearby_points), 4)  # Max 2 interceptors, but ensure we have points
        
        for i in range(interceptor_count):
            if i < len(nearby_points):
                launch_point = nearby_points[i][1]
                
                # Create interceptor jet
                interceptor_id = f"F16-{i+1:02d}"
                interceptor = EnhancedAircraft(
                    interceptor_id, launch_point, civilian_aircraft.current_navpoint, "FIGHTER_JET"
                )
                
                # Fighter jet characteristics
                interceptor.speed_kph = 1400  # Supersonic interception speed
                interceptor.altitude = civilian_aircraft.altitude + random.randint(-1000, 1000)  # Similar altitude
                
                # Try to calculate intercept path to civilian aircraft's current position
                path_created = False
                
                # Strategy 1: Try pathfinding to civilian's current navpoint
                if interceptor.calculate_path(self.current_airspace, civilian_aircraft.current_navpoint):
                    path_created = True
                
                # Strategy 2: Try pathfinding to civilian's destination
                elif interceptor.calculate_path(self.current_airspace, civilian_aircraft.end_navpoint):
                    path_created = True
                
                # Strategy 3: Create direct intercept path
                else:
                    # Create a simple direct path
                    interceptor.path_to_destination = [launch_point, civilian_aircraft.current_navpoint]
                    interceptor.current_navpoint = launch_point
                    interceptor.next_navpoint = civilian_aircraft.current_navpoint
                    interceptor.current_segment_progress = 0.0
                    interceptor.is_active = True
                    path_created = True
                    
                    if hasattr(self, 'advanced_info_text'):
                        self.advanced_info_text.insert(tk.END, 
                            f"⚠️ {interceptor_id} using direct intercept path\n")
                
                if path_created:
                    self.simulated_aircrafts.append(interceptor)
                    self.interception_scenario_data['interceptor_jets'].append(interceptor)
                    
                    if hasattr(self, 'advanced_info_text'):
                        self.advanced_info_text.insert(tk.END, 
                            f"🎯 {interceptor_id} launched from {launch_point.name} for intercept\n")
                else:
                    if hasattr(self, 'advanced_info_text'):
                        self.advanced_info_text.insert(tk.END, 
                            f"❌ Failed to create {interceptor_id} - no valid path\n")
        
        # If no interceptors were created, create at least one with fallback method
        if not self.interception_scenario_data['interceptor_jets']:
            if hasattr(self, 'advanced_info_text'):
                self.advanced_info_text.insert(tk.END, 
                    "🔄 Creating emergency interceptor with fallback method\n")
            
            # Use the closest navpoint regardless of pathfinding
            if nearby_points:
                launch_point = nearby_points[0][1]
                interceptor = EnhancedAircraft(
                    "F16-01", launch_point, civilian_aircraft.current_navpoint, "FIGHTER_JET"
                )
                interceptor.speed_kph = 1400
                interceptor.altitude = civilian_aircraft.altitude
                
                # Force create path
                interceptor.path_to_destination = [launch_point, civilian_aircraft.current_navpoint]
                interceptor.current_navpoint = launch_point
                interceptor.next_navpoint = civilian_aircraft.current_navpoint
                interceptor.current_segment_progress = 0.0
                interceptor.is_active = True
                
                self.simulated_aircrafts.append(interceptor)
                self.interception_scenario_data['interceptor_jets'].append(interceptor)
                
                if hasattr(self, 'advanced_info_text'):
                    self.advanced_info_text.insert(tk.END, 
                        f"✅ Emergency interceptor F16-01 created successfully\n")

    def update_interception_logic(self, delta_time_hours, current_time_epoch):
        """Update the interception scenario logic"""
        if not hasattr(self, 'interception_scenario_data'):
            return
            
        scenario = self.interception_scenario_data
        civilian_aircraft = scenario['civilian_aircraft']
        
        if not civilian_aircraft or not civilian_aircraft.is_active:
            return
        
        # Check if interception has been achieved
        if scenario['interception_triggered'] and not scenario['escort_phase']:
            civilian_lon, civilian_lat = civilian_aircraft.get_current_coordinates()
            
            # Check if any interceptor is within escort range
            for interceptor in scenario['interceptor_jets']:
                if interceptor.is_active:
                    int_lon, int_lat = interceptor.get_current_coordinates()
                    if int_lon and int_lat:
                        # Calculate distance between aircraft
                        dist_km = self.calculate_distance_km(
                            civilian_lon, civilian_lat, int_lon, int_lat
                        )
                        
                        if dist_km < 5:  # Within 5km - visual contact achieved
                            self.initiate_escort_phase(interceptor)
                            break

    def calculate_distance_km(self, lon1, lat1, lon2, lat2):
        """Calculate distance between two points using Haversine formula"""
        R = 6371  # Radius of Earth in kilometers
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad

        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def initiate_escort_phase(self, lead_interceptor):
        """Initiate the escort phase of interception"""
        self.interception_scenario_data['escort_phase'] = True
        
        if hasattr(self, 'advanced_info_text'):
            self.advanced_info_text.insert(tk.END, 
                f"🎯 VISUAL CONTACT ESTABLISHED - {lead_interceptor.aircraft_id}\n")
            self.advanced_info_text.insert(tk.END, 
                "🛡️ ESCORT PHASE INITIATED\n")
            self.advanced_info_text.insert(tk.END, 
                "📻 'CVL001, you are in restricted airspace. Follow our lead.'\n")
        
        # Modify civilian aircraft behavior - reduce speed and change course
        civilian_aircraft = self.interception_scenario_data['civilian_aircraft']
        civilian_aircraft.speed_kph = max(400, civilian_aircraft.speed_kph * 0.6)  # Reduce speed
        
        # Find nearest safe airport for escort
        self.escort_to_nearest_airport(civilian_aircraft, lead_interceptor)

    def escort_to_nearest_airport(self, civilian_aircraft, lead_interceptor):
        """Calculate escort route to nearest safe airport"""
        # Find airport navpoints
        airport_points = []
        for airport_name, airport in self.current_airspace.navairports.items():
            if airport.sids or airport.stars:
                airport_point = None
                if airport.sids:
                    airport_point = self.current_airspace.navpoints.get(airport.sids[0])
                elif airport.stars:
                    airport_point = self.current_airspace.navpoints.get(airport.stars[0])
                
                if airport_point:
                    airport_points.append((airport_name, airport_point))
        
        if airport_points:
            # Find nearest airport
            civilian_lon, civilian_lat = civilian_aircraft.get_current_coordinates()
            nearest_airport = min(airport_points, 
                key=lambda x: math.sqrt((x[1].longitude - civilian_lon)**2 + (x[1].latitude - civilian_lat)**2))
            
            airport_name, airport_point = nearest_airport
            
            # Reroute civilian aircraft to airport
            if civilian_aircraft.calculate_path(self.current_airspace, airport_point):
                if hasattr(self, 'advanced_info_text'):
                    self.advanced_info_text.insert(tk.END, 
                        f"🛬 Escorting CVL001 to {airport_name} airport\n")
            
            # Update interceptor to follow escort pattern
            lead_interceptor.calculate_path(self.current_airspace, airport_point)

    def export_interception_scenario_kml(self):
        """Export the complete interception scenario to KML with gx:Track"""
        if not hasattr(self, 'interception_scenario_data'):
            messagebox.showwarning("No Scenario", "No interception scenario has been run.")
            return
        
        if not any(aircraft.kml_track_data for aircraft in self.simulated_aircrafts):
            messagebox.showwarning("No Track Data", "No track data available. Run the scenario first.")
            return
        
        # Create comprehensive KML document
        kml_root = Element("kml", 
                          xmlns="http://www.opengis.net/kml/2.2",
                          attrib={"xmlns:gx": "http://www.google.com/kml/ext/2.2"})
        
        doc = SubElement(kml_root, "Document")
        SubElement(doc, "name").text = "Airspace Interception Scenario"
        SubElement(doc, "description").text = """
Complete interception scenario showing:
- Civilian aircraft violation of restricted airspace
- Fighter jet scramble and interception
- Escort to safe airport
- Real-time animated tracks with timestamps
        """
        
        # Add comprehensive styles
        self.add_interception_kml_styles(doc)
        
        # Create main scenario folder
        scenario_folder = SubElement(doc, "Folder")
        SubElement(scenario_folder, "name").text = "🎯 Interception Scenario"
        SubElement(scenario_folder, "open").text = "1"
        
        # Add restricted airspace zones
        zones_folder = SubElement(scenario_folder, "Folder")
        SubElement(zones_folder, "name").text = "🚫 Restricted Airspace"
        
        for zone in self.airspace_zones_3d:
            if zone.zone_type == 'RESTRICTED':
                zone_placemark = SubElement(zones_folder, "Placemark")
                SubElement(zone_placemark, "name").text = zone.name
                SubElement(zone_placemark, "description").text = f"""
<![CDATA[
<h3>{zone.name}</h3>
<p><b>Zone Type:</b> {zone.zone_type}</p>
<p><b>Altitude Range:</b> {zone.min_altitude_ft:,} - {zone.max_altitude_ft:,} ft</p>
<p><b>Description:</b> {zone.description}</p>
<p><b>Status:</b> ⚠️ VIOLATION DETECTED</p>
]]>
                """
                SubElement(zone_placemark, "styleUrl").text = "#restrictedZoneStyle"
                
                # Create 3D polygon
                polygon = SubElement(zone_placemark, "Polygon")
                SubElement(polygon, "extrude").text = "1"
                SubElement(polygon, "altitudeMode").text = "absolute"
                
                outer_boundary = SubElement(polygon, "outerBoundaryIs")
                linear_ring = SubElement(outer_boundary, "LinearRing")
                
                # Ensure polygon is closed and add altitude
                coords_3d = []
                for lon, lat in zone.vertices + [zone.vertices[0]]:
                    coords_3d.append(f"{lon},{lat},{zone.max_altitude_ft * 0.3048}")
                
                SubElement(linear_ring, "coordinates").text = " ".join(coords_3d)
        
        # Add aircraft tracks
        tracks_folder = SubElement(scenario_folder, "Folder")
        SubElement(tracks_folder, "name").text = "✈️ Aircraft Tracks"
        SubElement(tracks_folder, "open").text = "1"
        
        scenario = self.interception_scenario_data
        
        # Civilian aircraft track
        civilian_aircraft = scenario['civilian_aircraft']
        if civilian_aircraft and civilian_aircraft.kml_track_data:
            civilian_placemark = SubElement(tracks_folder, "Placemark")
            SubElement(civilian_placemark, "name").text = f"🛩️ {civilian_aircraft.aircraft_id} (Civilian)"
            
            violation_time = ""
            if scenario['interception_time']:
                violation_time = datetime.fromtimestamp(scenario['interception_time'], tz=timezone.utc).strftime("%H:%M:%S UTC")
            
            SubElement(civilian_placemark, "description").text = f"""
<![CDATA[
<h3>Civilian Aircraft - {civilian_aircraft.aircraft_id}</h3>
<p><b>Aircraft Type:</b> Commercial Airliner</p>
<p><b>Route:</b> {civilian_aircraft.start_navpoint.name} → {civilian_aircraft.end_navpoint.name}</p>
<p><b>Cruise Speed:</b> {civilian_aircraft.speed_kph} kph</p>
<p><b>Cruise Altitude:</b> FL{civilian_aircraft.altitude * 3.28084 / 100:.0f}</p>
<p><b>Violation Time:</b> {violation_time}</p>
<p><b>Status:</b> 🚨 RESTRICTED AIRSPACE VIOLATION</p>
]]>
            """
            SubElement(civilian_placemark, "styleUrl").text = "#civilianTrackStyle"
            
            # Create gx:Track for civilian aircraft
            gx_track = SubElement(civilian_placemark, "{http://www.google.com/kml/ext/2.2}Track")
            SubElement(gx_track, "altitudeMode").text = "absolute"
            
            for lon, lat, alt, timestamp in civilian_aircraft.kml_track_data:
                SubElement(gx_track, "when").text = timestamp
                SubElement(gx_track, "{http://www.google.com/kml/ext/2.2}coord").text = f"{lon} {lat} {alt}"
        
        # Interceptor jets tracks
        for i, interceptor in enumerate(scenario['interceptor_jets']):
            if interceptor.kml_track_data:
                interceptor_placemark = SubElement(tracks_folder, "Placemark")
                SubElement(interceptor_placemark, "name").text = f"🎯 {interceptor.aircraft_id} (Interceptor)"
                
                SubElement(interceptor_placemark, "description").text = f"""
<![CDATA[
<h3>Fighter Interceptor - {interceptor.aircraft_id}</h3>
<p><b>Aircraft Type:</b> Fighter Jet</p>
<p><b>Mission:</b> Intercept and Escort</p>
<p><b>Speed:</b> {interceptor.speed_kph} kph (Supersonic)</p>
<p><b>Altitude:</b> FL{interceptor.altitude * 3.28084 / 100:.0f}</p>
<p><b>Status:</b> 🎯 ACTIVE INTERCEPT</p>
]]>
                """
                SubElement(interceptor_placemark, "styleUrl").text = "#interceptorTrackStyle"
                
                # Create gx:Track for interceptor
                gx_track = SubElement(interceptor_placemark, "{http://www.google.com/kml/ext/2.2}Track")
                SubElement(gx_track, "altitudeMode").text = "absolute"
                
                for lon, lat, alt, timestamp in interceptor.kml_track_data:
                    SubElement(gx_track, "when").text = timestamp
                    SubElement(gx_track, "{http://www.google.com/kml/ext/2.2}coord").text = f"{lon} {lat} {alt}"
        
        # Add key events as placemarks
        events_folder = SubElement(scenario_folder, "Folder")
        SubElement(events_folder, "name").text = "📍 Key Events"
        
        if scenario['interception_triggered'] and civilian_aircraft:
            # Violation point
            violation_coords = None
            if civilian_aircraft.kml_track_data:
                # Find approximate violation point
                for i, (lon, lat, alt, timestamp) in enumerate(civilian_aircraft.kml_track_data):
                    event_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).timestamp()
                    if scenario['interception_time'] and abs(event_time - scenario['interception_time']) < 60:
                        violation_coords = (lon, lat, alt)
                        break
            
            if violation_coords:
                violation_placemark = SubElement(events_folder, "Placemark")
                SubElement(violation_placemark, "name").text = "🚨 Violation Point"
                SubElement(violation_placemark, "description").text = """
<![CDATA[
<h3>Restricted Airspace Violation</h3>
<p>This is where the civilian aircraft first entered restricted airspace, triggering the interception protocol.</p>
]]>
                """
                SubElement(violation_placemark, "styleUrl").text = "#violationPointStyle"
                
                point = SubElement(violation_placemark, "Point")
                SubElement(point, "coordinates").text = f"{violation_coords[0]},{violation_coords[1]},{violation_coords[2]}"
                SubElement(point, "altitudeMode").text = "absolute"
        
        # Convert to string and save
        file_path = filedialog.asksaveasfilename(
            defaultextension=".kml",
            filetypes=[("KML files", "*.kml")],
            title="Save Interception Scenario KML"
        )
        
        if file_path:
            try:
                # Pretty print the XML
                pretty_xml = minidom.parseString(tostring(kml_root, 'utf-8')).toprettyxml(indent="  ")
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(pretty_xml)
                
                messagebox.showinfo("KML Export Success", 
                    f"Complete interception scenario exported to:\n{file_path}\n\n"
                    f"Features included:\n"
                    f"• Animated aircraft tracks with timestamps\n"
                    f"• 3D restricted airspace zones\n"
                    f"• Key event markers\n"
                    f"• Comprehensive scenario documentation")
                
                self.open_kml_in_google_earth(file_path)
                
            except Exception as e:
                messagebox.showerror("KML Export Error", f"Failed to save KML file: {e}")

    def add_interception_kml_styles(self, doc):
        """Add comprehensive styles for the interception scenario KML"""
        
        # Restricted zone style
        restricted_style = SubElement(doc, "Style", id="restrictedZoneStyle")
        line_style = SubElement(restricted_style, "LineStyle")
        SubElement(line_style, "color").text = "ff0000ff"  # Red
        SubElement(line_style, "width").text = "4"
        poly_style = SubElement(restricted_style, "PolyStyle")
        SubElement(poly_style, "color").text = "4c0000ff"  # Semi-transparent red
        SubElement(poly_style, "fill").text = "1"
        SubElement(poly_style, "outline").text = "1"
        
        # Civilian aircraft track style
        civilian_style = SubElement(doc, "Style", id="civilianTrackStyle")
        icon_style = SubElement(civilian_style, "IconStyle")
        icon = SubElement(icon_style, "Icon")
        SubElement(icon, "href").text = "http://maps.google.com/mapfiles/kml/shapes/airports.png"
        SubElement(icon_style, "scale").text = "1.2"
        SubElement(icon_style, "color").text = "ff0080ff"  # Orange
        line_style = SubElement(civilian_style, "LineStyle")
        SubElement(line_style, "color").text = "ff0080ff"  # Orange
        SubElement(line_style, "width").text = "3"
        
        # Interceptor track style  
        interceptor_style = SubElement(doc, "Style", id="interceptorTrackStyle")
        icon_style = SubElement(interceptor_style, "IconStyle")
        icon = SubElement(icon_style, "Icon")
        SubElement(icon, "href").text = "http://maps.google.com/mapfiles/kml/shapes/target.png"
        SubElement(icon_style, "scale").text = "1.0"
        SubElement(icon_style, "color").text = "ff0000ff"  # Red
        line_style = SubElement(interceptor_style, "LineStyle")
        SubElement(line_style, "color").text = "ff0000ff"  # Red
        SubElement(line_style, "width").text = "2"
        
        # Violation point style
        violation_style = SubElement(doc, "Style", id="violationPointStyle")
        icon_style = SubElement(violation_style, "IconStyle")
        icon = SubElement(icon_style, "Icon")
        SubElement(icon, "href").text = "http://maps.google.com/mapfiles/kml/shapes/forbidden.png"
        SubElement(icon_style, "scale").text = "1.5"
        SubElement(icon_style, "color").text = "ff0000ff"  # Red
# Part 7: Main Execution Block

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = CompleteAirspaceInterface(root)
    root.mainloop()

if __name__ == "__main__":
    main()