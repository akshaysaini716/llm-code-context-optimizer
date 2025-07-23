# Simple Calculator - Test Project

A basic Python calculator application designed to test the **LLM Code Context Optimizer**.

## Project Structure

```
sample_project/
├── main.py              # Main application entry point
├── calculator.py        # Calculator class with arithmetic operations
├── file_handler.py      # File operations for saving/loading history
├── config.py           # Configuration management
├── test_calculator.py  # Unit tests
└── README.md           # This file
```

## Features

- **Basic arithmetic operations**: add, subtract, multiply, divide, power, square root
- **Calculation history**: Tracks all operations with timestamps
- **File persistence**: Save/load calculation history to JSON file
- **Configuration management**: Customizable settings via environment variables
- **Error handling**: Proper validation and error messages
- **Unit tests**: Comprehensive test coverage

## Usage

### Running the Calculator

```bash
cd sample_project
python main.py
```

### Available Commands

- `add` - Add two numbers
- `subtract` - Subtract two numbers  
- `multiply` - Multiply two numbers
- `divide` - Divide two numbers
- `history` - View calculation history
- `save` - Save history to file
- `load` - Load history from file
- `quit` - Exit the application

### Running Tests

```bash
python test_calculator.py
```

## Testing with LLM Code Context Optimizer

This project is perfect for testing the context optimizer with various scenarios:

### Test Scenarios

1. **Full Context Mode** - Include all files to understand the complete application
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{
       "message": "How does the calculator save history?",
       "include_context": true,
       "project_path": "/path/to/sample_project",
       "relevant_only": false
     }'
   ```

2. **Relevant Context Mode** - Only include files related to specific queries
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{
       "message": "Fix the division by zero error handling",
       "include_context": true,
       "project_path": "/path/to/sample_project",
       "relevant_only": true
     }'
   ```

3. **Configuration Questions** - Test context filtering for config-related queries
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{
       "message": "How can I change the history file location?",
       "include_context": true,
       "project_path": "/path/to/sample_project",
       "relevant_only": true
     }'
   ```

4. **Testing Questions** - Focus on test-related files
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{
       "message": "Add tests for the power function",
       "include_context": true,
       "project_path": "/path/to/sample_project",
       "relevant_only": true
     }'
   ```

### Expected Context Optimization Results

- **Query about "history"** → Should include `file_handler.py`, `calculator.py`, `main.py`
- **Query about "configuration"** → Should include `config.py`, possibly `main.py`
- **Query about "testing"** → Should include `test_calculator.py`, `calculator.py`
- **Query about "arithmetic"** → Should include `calculator.py`, possibly `test_calculator.py`
- **Query about "file operations"** → Should include `file_handler.py`

## Environment Variables

You can customize the calculator behavior using these environment variables:

- `CALC_HISTORY_FILE` - History file path (default: `calculator_history.json`)
- `CALC_MAX_HISTORY` - Maximum history entries (default: `50`)
- `CALC_DECIMAL_PLACES` - Decimal precision (default: `4`)
- `CALC_AUTO_SAVE` - Auto-save on exit (default: `true`)
- `CALC_BACKUP` - Enable backup creation (default: `false`)

## Example Usage

```python
from calculator import Calculator
from file_handler import FileHandler
from config import Config

# Initialize components
config = Config()
calc = Calculator()
file_handler = FileHandler(config.history_file)

# Perform calculations
result = calc.add(5, 3)  # Returns 8
result = calc.multiply(4, 7)  # Returns 28

# Save history
file_handler.save_history(calc.get_history())
```

This project provides a realistic codebase with multiple interconnected files, making it ideal for testing how well the LLM Code Context Optimizer can identify relevant code based on different types of queries. 