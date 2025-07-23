"""
Simple tests for the calculator application
"""

import unittest
import os
import tempfile
from calculator import Calculator
from file_handler import FileHandler
from config import Config

class TestCalculator(unittest.TestCase):
    """Test cases for the Calculator class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.calc = Calculator()
    
    def test_addition(self):
        """Test addition operation"""
        result = self.calc.add(2, 3)
        self.assertEqual(result, 5)
        
        result = self.calc.add(-1, 1)
        self.assertEqual(result, 0)
        
        result = self.calc.add(0.1, 0.2)
        self.assertAlmostEqual(result, 0.3, places=10)
    
    def test_subtraction(self):
        """Test subtraction operation"""
        result = self.calc.subtract(5, 3)
        self.assertEqual(result, 2)
        
        result = self.calc.subtract(0, 5)
        self.assertEqual(result, -5)
    
    def test_multiplication(self):
        """Test multiplication operation"""
        result = self.calc.multiply(3, 4)
        self.assertEqual(result, 12)
        
        result = self.calc.multiply(-2, 3)
        self.assertEqual(result, -6)
        
        result = self.calc.multiply(0, 100)
        self.assertEqual(result, 0)
    
    def test_division(self):
        """Test division operation"""
        result = self.calc.divide(10, 2)
        self.assertEqual(result, 5)
        
        result = self.calc.divide(7, 2)
        self.assertEqual(result, 3.5)
        
        # Test division by zero
        with self.assertRaises(ValueError):
            self.calc.divide(5, 0)
    
    def test_power(self):
        """Test power operation"""
        result = self.calc.power(2, 3)
        self.assertEqual(result, 8)
        
        result = self.calc.power(5, 0)
        self.assertEqual(result, 1)
    
    def test_square_root(self):
        """Test square root operation"""
        result = self.calc.square_root(9)
        self.assertEqual(result, 3)
        
        result = self.calc.square_root(0)
        self.assertEqual(result, 0)
        
        # Test negative number
        with self.assertRaises(ValueError):
            self.calc.square_root(-1)
    
    def test_history_tracking(self):
        """Test that operations are recorded in history"""
        self.calc.add(1, 2)
        self.calc.multiply(3, 4)
        
        history = self.calc.get_history()
        self.assertEqual(len(history), 2)
        self.assertIn("1 + 2 = 3", history[0])
        self.assertIn("3 × 4 = 12", history[1])
    
    def test_average_calculation(self):
        """Test average calculation"""
        numbers = [1, 2, 3, 4, 5]
        result = self.calc.calculate_average(numbers)
        self.assertEqual(result, 3.0)
        
        # Test empty list
        with self.assertRaises(ValueError):
            self.calc.calculate_average([])

class TestFileHandler(unittest.TestCase):
    """Test cases for the FileHandler class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Use temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        self.file_handler = FileHandler(self.temp_file.name)
    
    def tearDown(self):
        """Clean up after tests"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_save_and_load_history(self):
        """Test saving and loading history"""
        test_history = ["[2024-01-01 12:00:00] 2 + 3 = 5", "[2024-01-01 12:01:00] 4 × 5 = 20"]
        
        # Save history
        success = self.file_handler.save_history(test_history)
        self.assertTrue(success)
        
        # Load history
        loaded_history = self.file_handler.load_history()
        self.assertEqual(loaded_history, test_history)
    
    def test_file_info(self):
        """Test getting file information"""
        info = self.file_handler.get_file_info()
        self.assertTrue(info["exists"])
        self.assertIn("size_bytes", info)
        self.assertIn("modified_time", info)

class TestConfig(unittest.TestCase):
    """Test cases for the Config class"""
    
    def test_default_configuration(self):
        """Test default configuration values"""
        config = Config()
        
        self.assertEqual(config.app_name, "Simple Calculator")
        self.assertEqual(config.version, "1.0.0")
        self.assertIsInstance(config.decimal_places, int)
        self.assertIsInstance(config.auto_save, bool)
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        config = Config()
        validation = config.validate_settings()
        
        self.assertIn("valid", validation)
        self.assertIn("issues", validation)
        self.assertIn("settings", validation)
    
    def test_supported_operations(self):
        """Test getting supported operations"""
        config = Config()
        operations = config.get_supported_operations()
        
        self.assertIn("add", operations)
        self.assertIn("multiply", operations)
        self.assertIn("divide", operations)

def run_tests():
    """Run all tests"""
    unittest.main(verbosity=2)

if __name__ == "__main__":
    run_tests() 