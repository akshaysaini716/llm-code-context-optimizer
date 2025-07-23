#!/usr/bin/env python3
"""
Simple Calculator Application
A basic Python project to test the LLM Code Context Optimizer

This is a simple calculator that can:
- Perform basic arithmetic operations
- Save calculation history to file
- Load previous calculations
"""

from calculator import Calculator
from file_handler import FileHandler
from config import Config

def main():
    """Main entry point for the calculator application"""
    print("=== Simple Calculator ===")
    print("Commands: add, subtract, multiply, divide, history, save, load, quit")
    
    # Initialize components
    config = Config()
    file_handler = FileHandler(config.history_file)
    calculator = Calculator()
    
    # Load previous history
    history = file_handler.load_history()
    calculator.set_history(history)
    
    while True:
        try:
            command = input("\nEnter command: ").strip().lower()
            
            if command == "quit":
                # Save history before exiting
                file_handler.save_history(calculator.get_history())
                print("Goodbye!")
                break
            
            elif command == "add":
                a = float(input("Enter first number: "))
                b = float(input("Enter second number: "))
                result = calculator.add(a, b)
                print(f"Result: {result}")
            
            elif command == "subtract":
                a = float(input("Enter first number: "))
                b = float(input("Enter second number: "))
                result = calculator.subtract(a, b)
                print(f"Result: {result}")
            
            elif command == "multiply":
                a = float(input("Enter first number: "))
                b = float(input("Enter second number: "))
                result = calculator.multiply(a, b)
                print(f"Result: {result}")
            
            elif command == "divide":
                a = float(input("Enter first number: "))
                b = float(input("Enter second number: "))
                result = calculator.divide(a, b)
                print(f"Result: {result}")
            
            elif command == "history":
                history = calculator.get_history()
                if history:
                    print("\nCalculation History:")
                    for i, calc in enumerate(history, 1):
                        print(f"{i}. {calc}")
                else:
                    print("No history available")
            
            elif command == "save":
                file_handler.save_history(calculator.get_history())
                print("History saved successfully!")
            
            elif command == "load":
                history = file_handler.load_history()
                calculator.set_history(history)
                print(f"Loaded {len(history)} calculations from history")
            
            else:
                print("Unknown command. Available: add, subtract, multiply, divide, history, save, load, quit")
        
        except ValueError as e:
            print(f"Invalid input: {e}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()