"""
Complete Standalone Airspace Navigation Interface
Combines all functionalities from interface2.py and interface3.2.py
No external imports needed - all classes defined internally
"""
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import math


# ===== Core Graph Classes =====
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

# ===== Navigation Classes =====
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
    
    def find_shortest_path(self, start_id, end_id):
        """Find shortest path using Dijkstra's algorithm"""
        if start_id not in self.navpoints or end_id not in self.navpoints:
            return [], float('inf')
        
        # Initialize distances and previous nodes
        distances = {node_id: float('inf') for node_id in self.navpoints}
        distances[start_id] = 0
        previous = {}
        unvisited = set(self.navpoints.keys())
        
        while unvisited:
            # Find unvisited node with minimum distance
            current = min(unvisited, key=lambda x: distances[x])
            
            if distances[current] == float('inf'):
                break
            
            if current == end_id:
                break
            
            unvisited.remove(current)
            
            # Check neighbors
            for segment in self.navsegments:
                if segment.origin_number == current:
                    neighbor = segment.destination_number
                    if neighbor in unvisited:
                        alt_distance = distances[current] + segment.distance
                        if alt_distance < distances[neighbor]:
                            distances[neighbor] = alt_distance
                            previous[neighbor] = current
        
        # Reconstruct path
        if end_id not in previous and start_id != end_id:
            return [], float('inf')
        
        path = []
        current = end_id
        while current is not None:
            path.insert(0, current)
            current = previous.get(current)
        
        return path, distances[end_id]

# ===== Graph Algorithm Functions =====
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

# ===== File Parsing Functions =====
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
    
    def setup_visualization_area(self, parent, viz_type):
        """Setup visualization area for a tab with bigger graphs"""
        # Create bottom frame with visualization area and info panel
        bottom_frame = ttk.Frame(parent)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Graph visualization area - bigger and with cool styling
        viz_frame = ttk.LabelFrame(bottom_frame, text=f"{viz_type.title()} Visualization")
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure and canvas - increased size
        if viz_type == "airspace":
            self.airspace_figure = plt.Figure(figsize=(14, 10), dpi=100, facecolor=self.bg_very_light)
            self.airspace_figure.patch.set_facecolor(self.bg_very_light)
            
            self.airspace_canvas = FigureCanvasTkAgg(self.airspace_figure, viz_frame)
            self.airspace_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Add toolbar with cool styling
            toolbar_frame = ttk.Frame(viz_frame)
            toolbar_frame.pack(fill=tk.X, padx=5, pady=2)
            self.airspace_toolbar = NavigationToolbar2Tk(self.airspace_canvas, toolbar_frame)
            self.airspace_toolbar.update()
            
        else:
            self.graph_figure = plt.Figure(figsize=(14, 10), dpi=100, facecolor=self.bg_very_light)
            self.graph_figure.patch.set_facecolor(self.bg_very_light)
            
            self.graph_canvas = FigureCanvasTkAgg(self.graph_figure, viz_frame)
            self.graph_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Add toolbar with cool styling
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
        else:
            self.graph_info_text = tk.Text(info_frame, height=6, wrap=tk.WORD,
                                         bg=self.bg_very_light, fg=self.text_dark,
                                         font=("Consolas", 9), relief="flat",
                                         borderwidth=1, highlightthickness=1,
                                         highlightcolor=self.accent_cyan)
            self.graph_info_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
    
    # ===== Airspace Functions =====
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
                except:
                    print("Could not list directory contents")
                
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
            messagebox.showwarning("No Airspace", "Please load an airspace first.")
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
            if airport.sids:
                sid_point = self.current_airspace.navpoints.get(airport.sids[0])
                if sid_point:
                    ax.scatter(sid_point.longitude, sid_point.latitude, s=120, c='#e53e3e', marker='^', zorder=3, alpha=0.9)
                    ax.text(sid_point.longitude, sid_point.latitude + 0.01, airport_name, 
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
            lat_padding = (max(lats) - min(lats)) * 0.05
            lon_padding = (max(lons) - min(lons)) * 0.05
            ax.set_xlim(min(lons) - lon_padding, max(lons) + lon_padding)
            ax.set_ylim(min(lats) - lat_padding, max(lats) + lat_padding)
        
        self.airspace_canvas.draw()
        
        # Update info with cool styling and emojis
        self.airspace_info_text.delete(1.0, tk.END)
        self.airspace_info_text.insert(tk.END, f"🌍 Airspace: {self.current_airspace_name.get()}\n")
        self.airspace_info_text.insert(tk.END, f"📍 Navigation Points: {len(self.current_airspace.navpoints)}\n")
        self.airspace_info_text.insert(tk.END, f"🔗 Segments: {len(self.current_airspace.navsegments)}\n")
        self.airspace_info_text.insert(tk.END, f"✈️ Airports: {len(self.current_airspace.navairports)}\n")
        self.airspace_info_text.insert(tk.END, f"🎨 Clean visualization with arrows, distances, and node names\n")
        
        print(f"Visualization complete: {points_plotted} points, {segments_plotted} segments, {airports_plotted} airports")
    
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
            return
        
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
        
        # Plot airport location (use first SID)
        airport_point = None
        if airport.sids:
            airport_point = self.current_airspace.navpoints.get(airport.sids[0])
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
    
    # ===== Graph Editor Functions =====
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
            
            # Remove from neighbor lists
            node_to_delete = self.graph.nodes[node_name]
            for other_node in self.graph.nodes.values():
                if node_to_delete in other_node.neighbors:
                    other_node.neighbors.remove(node_to_delete)
            
            # Remove the node
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


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = CompleteAirspaceInterface(root)
    root.mainloop()

if __name__ == "__main__":
    main()