import unittest
import os
import tempfile
from airSpace import AirSpace
from navPoint import NavPoint
from navSegment import NavSegment
from navAirport import NavAirport

class TestAirSpace(unittest.TestCase):
    """
    Test class for the AirSpace class and its functions.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        self.airspace = AirSpace("Test Airspace")
        
        # Create test data
        # Navigation Points
        self.navpoints = {
            1: NavPoint(1, "Point1", 41.0, 2.0),
            2: NavPoint(2, "Point2", 41.1, 2.1),
            3: NavPoint(3, "Point3", 41.2, 2.2),
            4: NavPoint(4, "Point4", 41.3, 2.3),
            5: NavPoint(5, "Point5", 41.4, 2.4)
        }
        
        # Segments
        self.segments = [
            NavSegment(1, 2, 10),
            NavSegment(2, 3, 15),
            NavSegment(3, 4, 5),
            NavSegment(4, 5, 7),
            NavSegment(1, 3, 20),
            NavSegment(2, 4, 18)
        ]
        
        # Airports
        self.airports = {
            "Airport1": NavAirport("Airport1", [1], [5]),
            "Airport2": NavAirport("Airport2", [2], [4])
        }
        
        # Set up the airspace with test data
        self.airspace.navpoints = self.navpoints
        self.airspace.navsegments = self.segments
        self.airspace.navairports = self.airports
    
    def test_init(self):
        """
        Test initialization of AirSpace.
        """
        airspace = AirSpace("Test")
        self.assertEqual(airspace.name, "Test")
        self.assertEqual(len(airspace.navpoints), 0)
        self.assertEqual(len(airspace.navsegments), 0)
        self.assertEqual(len(airspace.navairports), 0)
    
    def test_str(self):
        """
        Test string representation of AirSpace.
        """
        expected_str = "AirSpace: Test Airspace (Points: 5, Segments: 6, Airports: 2)"
        self.assertEqual(str(self.airspace), expected_str)
    
    def test_get_info(self):
        """
        Test get_info method of AirSpace.
        """
        expected_info = ("AirSpace: Test Airspace\n"
                          "Navigation Points: 5\n"
                          "Navigation Segments: 6\n"
                          "Airports: 2\n")
        self.assertEqual(self.airspace.get_info(), expected_info)
    
    def test_load_from_files(self):
        """
        Test loading airspace data from files.
        """
        # Create temporary test files
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as nav_file:
            nav_file.write("1 Point1 41.0 2.0\n")
            nav_file.write("2 Point2 41.1 2.1\n")
            nav_file_name = nav_file.name
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as seg_file:
            seg_file.write("1 2 10\n")
            seg_file.write("2 1 10\n")
            seg_file_name = seg_file.name
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as aer_file:
            aer_file.write("Airport1\n")
            aer_file.write("1 ...SID\n")
            aer_file.write("2 ...STAR\n")
            aer_file_name = aer_file.name
        
        # Create a new airspace
        airspace = AirSpace("Test Airspace")
        
        # Load data from files
        success = airspace.load_from_files(nav_file_name, seg_file_name, aer_file_name)
        
        # Check if loading was successful
        self.assertTrue(success)
        
        # Check if the data was loaded correctly
        self.assertEqual(len(airspace.navpoints), 2)
        self.assertEqual(len(airspace.navsegments), 2)
        self.assertGreaterEqual(len(airspace.navairports), 1)
        
        # Clean up the temporary files
        os.unlink(nav_file_name)
        os.unlink(seg_file_name)
        os.unlink(aer_file_name)
        
        # Test loading from non-existent files
        airspace = AirSpace("Test Airspace")
        success = airspace.load_from_files("non_existent_file1.txt", "non_existent_file2.txt", "non_existent_file3.txt")
        self.assertFalse(success)
    
    def test_get_navpoint_and_airport(self):
        """
        Test getting navigation points and airports.
        """
        # Test getting a navigation point that exists
        navpoint = self.airspace.get_navpoint(1)
        self.assertIsNotNone(navpoint)
        self.assertEqual(navpoint.name, "Point1")
        
        # Test getting a navigation point that doesn't exist
        navpoint = self.airspace.get_navpoint(999)
        self.assertIsNone(navpoint)
        
        # Test getting an airport that exists
        airport = self.airspace.get_airport("Airport1")
        self.assertIsNotNone(airport)
        self.assertEqual(airport.name, "Airport1")
        
        # Test getting an airport that doesn't exist
        airport = self.airspace.get_airport("NonExistentAirport")
        self.assertIsNone(airport)
    
    def test_get_neighbors(self):
        """
        Test getting neighbors of a navigation point.
        """
        # Test getting neighbors of point 1
        neighbors = self.airspace.get_neighbors(1)
        self.assertEqual(len(neighbors), 2)
        
        # Check if the correct neighbors were found
        neighbor_names = [n.name for n in neighbors]
        self.assertIn("Point2", neighbor_names)
        self.assertIn("Point3", neighbor_names)
        
        # Test getting neighbors of a point that doesn't exist
        neighbors = self.airspace.get_neighbors(999)
        self.assertEqual(len(neighbors), 0)
    
    def test_get_segment(self):
        """
        Test getting a specific segment.
        """
        # Test getting a segment that exists
        segment = self.airspace.get_segment(1, 2)
        self.assertIsNotNone(segment)
        self.assertEqual(segment.distance, 10)
        
        # Test getting a segment that doesn't exist
        segment = self.airspace.get_segment(1, 5)
        self.assertIsNone(segment)
    
    def test_find_shortest_path(self):
        """
        Test finding the shortest path between two points.
        """
        # Test finding the shortest path from point 1 to point 5
        path, cost = self.airspace.find_shortest_path(1, 5)
        
        # Check if a path was found
        self.assertGreater(len(path), 0)
        
        # Check if the path starts with the origin and ends with the destination
        self.assertEqual(path[0], 1)
        self.assertEqual(path[-1], 5)
        
        # Check that the cost is reasonable
        self.assertLess(cost, 100)
        
        # Test finding a path between points that are not connected
        # Create a new isolated point
        self.airspace.navpoints[6] = NavPoint(6, "IsolatedPoint", 42.0, 3.0)
        
        path, cost = self.airspace.find_shortest_path(1, 6)
        
        # Check that no path was found
        self.assertEqual(len(path), 0)
        self.assertEqual(cost, float('inf'))
    
    def test_find_reachable_points(self):
        """
        Test finding reachable points from a given point.
        """
        # Test finding points reachable from point 1
        reachable = self.airspace.find_reachable_points(1)
        
        # Check if the correct points are reachable
        self.assertEqual(len(reachable), 4)  # All other points should be reachable
        self.assertIn(2, reachable)
        self.assertIn(3, reachable)
        self.assertIn(4, reachable)
        self.assertIn(5, reachable)
        
        # Test finding points reachable from a point that doesn't exist
        reachable = self.airspace.find_reachable_points(999)
        self.assertEqual(len(reachable), 0)
        
        # Test finding points reachable from an isolated point
        # Create a new isolated point
        self.airspace.navpoints[6] = NavPoint(6, "IsolatedPoint", 42.0, 3.0)
        
        reachable = self.airspace.find_reachable_points(6)
        self.assertEqual(len(reachable), 0)

if __name__ == '__main__':
    unittest.main()