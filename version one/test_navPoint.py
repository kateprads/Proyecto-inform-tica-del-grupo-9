import unittest
from navPoint import NavPoint, read_navpoints_from_file, calculate_distance

class TestNavPoint(unittest.TestCase):
    """
    Test class for the NavPoint class and its functions.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        self.test_point = NavPoint(6063, "IZA.D", 38.8731546833, 1.37242975)
    
    def test_init(self):
        """
        Test initialization of NavPoint.
        """
        self.assertEqual(self.test_point.number, 6063)
        self.assertEqual(self.test_point.name, "IZA.D")
        self.assertEqual(self.test_point.latitude, 38.8731546833)
        self.assertEqual(self.test_point.longitude, 1.37242975)
    
    def test_str(self):
        """
        Test string representation of NavPoint.
        """
        expected_str = "6063 IZA.D (38.8731546833, 1.37242975)"
        self.assertEqual(str(self.test_point), expected_str)
    
    def test_get_info(self):
        """
        Test get_info method of NavPoint.
        """
        expected_info = ("Navigation Point #6063\n"
                          "Name: IZA.D\n"
                          "Coordinates: (38.8731546833, 1.37242975)")
        self.assertEqual(self.test_point.get_info(), expected_info)
    
    def test_calculate_distance(self):
        """
        Test distance calculation between two points.
        """
        point1 = NavPoint(1, "Point1", 41.0, 2.0)
        point2 = NavPoint(2, "Point2", 41.1, 2.1)
        
        # Calculate the expected distance using the Haversine formula
        distance = calculate_distance(point1, point2)
        
        # The distance should be approximately 15.4 km
        self.assertAlmostEqual(distance, 15.4, delta=0.5)
    
    def test_read_navpoints_from_file(self):
        """
        Test reading navigation points from a file.
        """
        # Create a temporary test file
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write("......\n")
            temp_file.write("4954 GIR 41.9313888889 2.7716666667\n")
            temp_file.write("4955 GIR.A 41.9022670667 2.7603478667\n")
            temp_file.write("5129 GODOX 39.3725 1.4108333333\n")
            temp_file_name = temp_file.name
        
        # Read the navigation points from the test file
        navpoints = read_navpoints_from_file(temp_file_name)
        
        # Check if the correct number of points were read
        self.assertEqual(len(navpoints), 3)
        
        # Check if the points have the correct values
        self.assertEqual(navpoints[4954].name, "GIR")
        self.assertEqual(navpoints[4955].name, "GIR.A")
        self.assertEqual(navpoints[5129].name, "GODOX")
        
        # Test reading a non-existent file
        navpoints = read_navpoints_from_file("non_existent_file.txt")
        self.assertEqual(len(navpoints), 0)
        
        # Clean up the temporary file
        import os
        os.unlink(temp_file_name)

if __name__ == '__main__':
    unittest.main()