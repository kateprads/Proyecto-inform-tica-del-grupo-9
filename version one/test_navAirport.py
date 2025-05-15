import unittest
from navAirport import NavAirport, parse_airport_file

class TestNavAirport(unittest.TestCase):
    """
    Test class for the NavAirport class and its functions.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        self.test_airport = NavAirport("LEIB", [6063], [6061])
    
    def test_init(self):
        """
        Test initialization of NavAirport.
        """
        self.assertEqual(self.test_airport.name, "LEIB")
        self.assertEqual(self.test_airport.sids, [6063])
        self.assertEqual(self.test_airport.stars, [6061])
    
    def test_str(self):
        """
        Test string representation of NavAirport.
        """
        expected_str = "Airport: LEIB (SIDs: 1, STARs: 1)"
        self.assertEqual(str(self.test_airport), expected_str)
    
    def test_get_info(self):
        """
        Test get_info method of NavAirport.
        """
        expected_info = ("Airport: LEIB\n"
                          "SIDs (Departure points):\n"
                          "  6063\n"
                          "STARs (Arrival points):\n"
                          "  6061\n")
        self.assertEqual(self.test_airport.get_info(), expected_info)
    
    def test_add_sid_and_star(self):
        """
        Test adding SIDs and STARs.
        """
        # Create a new airport
        airport = NavAirport("LEBL")
        
        # Add SIDs and STARs
        airport.add_sid(5129)
        airport.add_sid(5335)
        airport.add_star(6063)
        
        # Check if the SIDs and STARs were added
        self.assertEqual(len(airport.sids), 2)
        self.assertEqual(len(airport.stars), 1)
        self.assertEqual(airport.sids[0], 5129)
        self.assertEqual(airport.sids[1], 5335)
        self.assertEqual(airport.stars[0], 6063)
        
        # Test adding a duplicate SID and STAR
        airport.add_sid(5129)
        airport.add_star(6063)
        
        # Check that duplicates are not added
        self.assertEqual(len(airport.sids), 2)
        self.assertEqual(len(airport.stars), 1)
    
    def test_get_first_sid(self):
        """
        Test getting the first SID.
        """
        # Create airports with and without SIDs
        airport_with_sids = NavAirport("LEBL", [5129, 5335])
        airport_without_sids = NavAirport("LEGE")
        
        # Test getting the first SID
        self.assertEqual(airport_with_sids.get_first_sid(), 5129)
        self.assertIsNone(airport_without_sids.get_first_sid())
    
    def test_parse_airport_file(self):
        """
        Test parsing airport file.
        """
        # Create a temporary test file
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write("......\n")
            temp_file.write("LEIB\n")
            temp_file.write("6063 IZA.D ...other departures (SIDs)\n")
            temp_file.write("6061 IZA.A ...other arrivals (STARs)\n")
            temp_file.write("\n")
            temp_file.write("LEBL\n")
            temp_file.write("5129 GODOX ...other departures (SIDs)\n")
            temp_file.write("5335 GRAUS ...other arrivals (STARs)\n")
            temp_file_name = temp_file.name
        
        # Parse the airport file
        airports = parse_airport_file(temp_file_name)
        
        # Check if the correct number of airports were parsed
        self.assertEqual(len(airports), 2)
        
        # Check if the airports have the correct values
        self.assertIn("LEIB", airports)
        self.assertIn("LEBL", airports)
        
        # Parse a non-existent file
        airports = parse_airport_file("non_existent_file.txt")
        self.assertEqual(len(airports), 0)
        
        # Clean up the temporary file
        import os
        os.unlink(temp_file_name)

if __name__ == '__main__':
    unittest.main()