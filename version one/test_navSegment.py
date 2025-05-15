import unittest
from navSegment import NavSegment, read_segments_from_file, get_segments_by_origin, get_segment

class TestNavSegment(unittest.TestCase):
    """
    Test class for the NavSegment class and its functions.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        self.test_segment = NavSegment(6063, 6937, 48.55701)
    
    def test_init(self):
        """
        Test initialization of NavSegment.
        """
        self.assertEqual(self.test_segment.origin_number, 6063)
        self.assertEqual(self.test_segment.destination_number, 6937)
        self.assertEqual(self.test_segment.distance, 48.55701)
    
    def test_str(self):
        """
        Test string representation of NavSegment.
        """
        expected_str = "Segment: 6063 -> 6937 (Distance: 48.55701 km)"
        self.assertEqual(str(self.test_segment), expected_str)
    
    def test_get_info(self):
        """
        Test get_info method of NavSegment.
        """
        expected_info = ("Segment from 6063 to 6937\n"
                          "Distance: 48.55701 km")
        self.assertEqual(self.test_segment.get_info(), expected_info)
    
    def test_read_segments_from_file(self):
        """
        Test reading segments from a file.
        """
        # Create a temporary test file
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write("......\n")
            temp_file.write("5129 5535 26.183\n")
            temp_file.write("5129 11733 53.80333\n")
            temp_file.write("5335 7484 26.1183\n")
            temp_file.write("6063 6937 48.55701\n")
            temp_file_name = temp_file.name
        
        # Read the segments from the test file
        segments = read_segments_from_file(temp_file_name)
        
        # Check if the correct number of segments were read
        self.assertEqual(len(segments), 4)
        
        # Check if the segments have the correct values
        self.assertEqual(segments[0].origin_number, 5129)
        self.assertEqual(segments[0].destination_number, 5535)
        self.assertEqual(segments[0].distance, 26.183)
        
        self.assertEqual(segments[3].origin_number, 6063)
        self.assertEqual(segments[3].destination_number, 6937)
        self.assertEqual(segments[3].distance, 48.55701)
        
        # Test reading a non-existent file
        segments = read_segments_from_file("non_existent_file.txt")
        self.assertEqual(len(segments), 0)
        
        # Clean up the temporary file
        import os
        os.unlink(temp_file_name)
    
    def test_get_segments_by_origin(self):
        """
        Test getting segments by origin.
        """
        # Create some test segments
        segments = [
            NavSegment(1, 2, 10),
            NavSegment(1, 3, 15),
            NavSegment(2, 3, 5),
            NavSegment(3, 4, 7)
        ]
        
        # Get segments with origin 1
        origin_segments = get_segments_by_origin(segments, 1)
        
        # Check if the correct segments were found
        self.assertEqual(len(origin_segments), 2)
        self.assertEqual(origin_segments[0].destination_number, 2)
        self.assertEqual(origin_segments[1].destination_number, 3)
        
        # Get segments with origin 4 (should be none)
        origin_segments = get_segments_by_origin(segments, 4)
        self.assertEqual(len(origin_segments), 0)
    
    def test_get_segment(self):
        """
        Test getting a specific segment.
        """
        # Create some test segments
        segments = [
            NavSegment(1, 2, 10),
            NavSegment(1, 3, 15),
            NavSegment(2, 3, 5),
            NavSegment(3, 4, 7)
        ]
        
        # Get segment from 1 to 3
        segment = get_segment(segments, 1, 3)
        self.assertIsNotNone(segment)
        self.assertEqual(segment.distance, 15)
        
        # Get non-existent segment
        segment = get_segment(segments, 1, 4)
        self.assertIsNone(segment)

if __name__ == '__main__':
    unittest.main()