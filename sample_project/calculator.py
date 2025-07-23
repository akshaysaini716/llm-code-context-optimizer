"""
Calculator module for basic arithmetic operations
"""

from typing import List
from datetime import datetime

class Calculator:
    """A simple calculator class that performs basic arithmetic operations"""
    
    def __init__(self):
        """Initialize calculator with empty history"""
        self.history = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers and record the operation"""
        result = a + b
        self._record_operation(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract second number from first and record the operation"""
        result = a - b
        self._record_operation(f"{a} - {b} = {result}")
        return result
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers and record the operation"""
        result = a * b
        self._record_operation(f"{a} × {b} = {result}")
        return result
    
    def divide(self, a: float, b: float) -> float:
        """Divide first number by second and record the operation"""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        
        result = a / b
        self._record_operation(f"{a} ÷ {b} = {result}")
        return result
    
    def power(self, base: float, exponent: float) -> float:
        """Raise base to the power of exponent"""
        result = base ** exponent
        self._record_operation(f"{base} ^ {exponent} = {result}")
        return result
    
    def square_root(self, number: float) -> float:
        """Calculate square root of a number"""
        if number < 0:
            raise ValueError("Cannot calculate square root of negative number")
        
        result = number ** 0.5
        self._record_operation(f"√{number} = {result}")
        return result
    
    def _record_operation(self, operation: str):
        """Record an operation in the history with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.history.append(f"[{timestamp}] {operation}")
        
        # Keep only last 50 operations to prevent memory issues
        if len(self.history) > 50:
            self.history = self.history[-50:]
    
    def get_history(self) -> List[str]:
        """Get the calculation history"""
        return self.history.copy()
    
    def set_history(self, history: List[str]):
        """Set the calculation history"""
        self.history = history.copy()
    
    def clear_history(self):
        """Clear the calculation history"""
        self.history.clear()
    
    def get_last_result(self) -> str:
        """Get the last calculation result"""
        if self.history:
            return self.history[-1]
        return "No calculations performed yet"
    
    def calculate_average(self, numbers: List[float]) -> float:
        """Calculate the average of a list of numbers"""
        if not numbers:
            raise ValueError("Cannot calculate average of empty list")
        
        result = sum(numbers) / len(numbers)
        self._record_operation(f"Average of {numbers} = {result}")
        return result 