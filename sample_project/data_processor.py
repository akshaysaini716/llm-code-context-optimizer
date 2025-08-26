"""
Data Processing Module
Handles data transformation, validation, and analysis operations
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import csv
from datetime import datetime, timedelta
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Result of a data processing operation"""
    success: bool
    data: Optional[Any] = None
    errors: List[str] = None
    warnings: List[str] = None
    processing_time: float = 0.0
    records_processed: int = 0
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """Validate phone number format"""
        import re
        # Simple phone validation - can be enhanced
        pattern = r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$'
        return bool(re.match(pattern, phone.replace(' ', '')))
    
    @staticmethod
    def validate_date_range(date_str: str, min_date: str = None, max_date: str = None) -> bool:
        """Validate date is within specified range"""
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            
            if min_date:
                min_obj = datetime.strptime(min_date, '%Y-%m-%d')
                if date_obj < min_obj:
                    return False
            
            if max_date:
                max_obj = datetime.strptime(max_date, '%Y-%m-%d')
                if date_obj > max_obj:
                    return False
            
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_numeric_range(value: Union[int, float], min_val: float = None, max_val: float = None) -> bool:
        """Validate numeric value is within range"""
        try:
            num_val = float(value)
            
            if min_val is not None and num_val < min_val:
                return False
            
            if max_val is not None and num_val > max_val:
                return False
            
            return True
        except (ValueError, TypeError):
            return False

class DataTransformer(ABC):
    """Abstract base class for data transformers"""
    
    @abstractmethod
    def transform(self, data: Any) -> Any:
        """Transform the input data"""
        pass
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate the input data"""
        pass

class TextTransformer(DataTransformer):
    """Text data transformation utilities"""
    
    def __init__(self, lowercase: bool = True, strip_whitespace: bool = True, 
                 remove_special_chars: bool = False):
        self.lowercase = lowercase
        self.strip_whitespace = strip_whitespace
        self.remove_special_chars = remove_special_chars
    
    def transform(self, data: str) -> str:
        """Transform text data"""
        if not isinstance(data, str):
            return str(data)
        
        result = data
        
        if self.strip_whitespace:
            result = result.strip()
        
        if self.lowercase:
            result = result.lower()
        
        if self.remove_special_chars:
            import re
            result = re.sub(r'[^a-zA-Z0-9\s]', '', result)
        
        return result
    
    def validate(self, data: str) -> bool:
        """Validate text data"""
        return isinstance(data, str) and len(data.strip()) > 0
    
    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """Extract keywords from text"""
        words = text.lower().split()
        keywords = [word.strip('.,!?;:"()[]{}') for word in words]
        return [word for word in keywords if len(word) >= min_length]
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using simple word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

class NumericTransformer(DataTransformer):
    """Numeric data transformation utilities"""
    
    def __init__(self, normalize: bool = False, round_digits: Optional[int] = None):
        self.normalize = normalize
        self.round_digits = round_digits
        self._min_val = None
        self._max_val = None
    
    def fit(self, data: List[Union[int, float]]):
        """Fit the transformer to data (for normalization)"""
        if self.normalize:
            self._min_val = min(data)
            self._max_val = max(data)
    
    def transform(self, data: Union[int, float]) -> float:
        """Transform numeric data"""
        try:
            result = float(data)
            
            if self.normalize and self._min_val is not None and self._max_val is not None:
                if self._max_val != self._min_val:
                    result = (result - self._min_val) / (self._max_val - self._min_val)
            
            if self.round_digits is not None:
                result = round(result, self.round_digits)
            
            return result
        except (ValueError, TypeError):
            return 0.0
    
    def validate(self, data: Union[int, float]) -> bool:
        """Validate numeric data"""
        try:
            float(data)
            return True
        except (ValueError, TypeError):
            return False
    
    def calculate_statistics(self, data: List[Union[int, float]]) -> Dict[str, float]:
        """Calculate basic statistics"""
        if not data:
            return {}
        
        numeric_data = [self.transform(x) for x in data if self.validate(x)]
        
        if not numeric_data:
            return {}
        
        return {
            'mean': np.mean(numeric_data),
            'median': np.median(numeric_data),
            'std': np.std(numeric_data),
            'min': np.min(numeric_data),
            'max': np.max(numeric_data),
            'count': len(numeric_data)
        }

class DateTimeTransformer(DataTransformer):
    """DateTime data transformation utilities"""
    
    def __init__(self, input_format: str = '%Y-%m-%d', output_format: str = '%Y-%m-%d'):
        self.input_format = input_format
        self.output_format = output_format
    
    def transform(self, data: str) -> str:
        """Transform datetime string"""
        try:
            dt = datetime.strptime(data, self.input_format)
            return dt.strftime(self.output_format)
        except ValueError:
            return data
    
    def validate(self, data: str) -> bool:
        """Validate datetime string"""
        try:
            datetime.strptime(data, self.input_format)
            return True
        except ValueError:
            return False
    
    def parse_to_datetime(self, data: str) -> Optional[datetime]:
        """Parse string to datetime object"""
        try:
            return datetime.strptime(data, self.input_format)
        except ValueError:
            return None
    
    def calculate_age(self, birth_date: str, reference_date: str = None) -> Optional[int]:
        """Calculate age from birth date"""
        birth_dt = self.parse_to_datetime(birth_date)
        if not birth_dt:
            return None
        
        if reference_date:
            ref_dt = self.parse_to_datetime(reference_date)
        else:
            ref_dt = datetime.now()
        
        if not ref_dt:
            return None
        
        age = ref_dt.year - birth_dt.year
        if ref_dt.month < birth_dt.month or (ref_dt.month == birth_dt.month and ref_dt.day < birth_dt.day):
            age -= 1
        
        return age

class DataProcessor:
    """Main data processing class"""
    
    def __init__(self):
        self.transformers: Dict[str, DataTransformer] = {}
        self.validators: Dict[str, Callable] = {}
        self.processing_history: List[Dict] = []
    
    def add_transformer(self, name: str, transformer: DataTransformer):
        """Add a data transformer"""
        self.transformers[name] = transformer
    
    def add_validator(self, name: str, validator: Callable):
        """Add a data validator"""
        self.validators[name] = validator
    
    def process_record(self, record: Dict[str, Any], 
                      field_transformers: Dict[str, str] = None,
                      field_validators: Dict[str, str] = None) -> ProcessingResult:
        """
        Process a single data record
        
        Args:
            record: Dictionary containing the data record
            field_transformers: Mapping of field -> transformer name
            field_validators: Mapping of field -> validator name
            
        Returns:
            ProcessingResult with processed data and any errors
        """
        start_time = datetime.now()
        result = ProcessingResult(success=True)
        processed_record = record.copy()
        
        # Apply transformers
        if field_transformers:
            for field, transformer_name in field_transformers.items():
                if field in processed_record and transformer_name in self.transformers:
                    try:
                        transformer = self.transformers[transformer_name]
                        processed_record[field] = transformer.transform(processed_record[field])
                    except Exception as e:
                        result.errors.append(f"Transformation error for field {field}: {str(e)}")
                        result.success = False
        
        # Apply validators
        if field_validators:
            for field, validator_name in field_validators.items():
                if field in processed_record and validator_name in self.validators:
                    try:
                        validator = self.validators[validator_name]
                        if not validator(processed_record[field]):
                            result.errors.append(f"Validation failed for field {field}")
                            result.success = False
                    except Exception as e:
                        result.errors.append(f"Validation error for field {field}: {str(e)}")
                        result.success = False
        
        result.data = processed_record
        result.processing_time = (datetime.now() - start_time).total_seconds()
        result.records_processed = 1
        
        return result
    
    def process_batch(self, records: List[Dict[str, Any]], 
                     field_transformers: Dict[str, str] = None,
                     field_validators: Dict[str, str] = None,
                     parallel: bool = False) -> ProcessingResult:
        """
        Process a batch of data records
        
        Args:
            records: List of data records
            field_transformers: Mapping of field -> transformer name
            field_validators: Mapping of field -> validator name
            parallel: Whether to process records in parallel
            
        Returns:
            ProcessingResult with processed data and aggregated errors
        """
        start_time = datetime.now()
        
        if parallel:
            return self._process_batch_parallel(records, field_transformers, field_validators, start_time)
        else:
            return self._process_batch_sequential(records, field_transformers, field_validators, start_time)
    
    def _process_batch_sequential(self, records: List[Dict[str, Any]], 
                                field_transformers: Dict[str, str],
                                field_validators: Dict[str, str],
                                start_time: datetime) -> ProcessingResult:
        """Process batch sequentially"""
        processed_records = []
        all_errors = []
        all_warnings = []
        
        for i, record in enumerate(records):
            result = self.process_record(record, field_transformers, field_validators)
            
            if result.success:
                processed_records.append(result.data)
            else:
                all_errors.extend([f"Record {i}: {error}" for error in result.errors])
            
            all_warnings.extend([f"Record {i}: {warning}" for warning in result.warnings])
        
        return ProcessingResult(
            success=len(all_errors) == 0,
            data=processed_records,
            errors=all_errors,
            warnings=all_warnings,
            processing_time=(datetime.now() - start_time).total_seconds(),
            records_processed=len(records)
        )
    
    def _process_batch_parallel(self, records: List[Dict[str, Any]], 
                              field_transformers: Dict[str, str],
                              field_validators: Dict[str, str],
                              start_time: datetime) -> ProcessingResult:
        """Process batch in parallel"""
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Create partial function with fixed parameters
            process_func = functools.partial(
                self.process_record,
                field_transformers=field_transformers,
                field_validators=field_validators
            )
            
            # Submit all tasks
            futures = [executor.submit(process_func, record) for record in records]
            
            # Collect results
            processed_records = []
            all_errors = []
            all_warnings = []
            
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    
                    if result.success:
                        processed_records.append(result.data)
                    else:
                        all_errors.extend([f"Record {i}: {error}" for error in result.errors])
                    
                    all_warnings.extend([f"Record {i}: {warning}" for warning in result.warnings])
                
                except Exception as e:
                    all_errors.append(f"Record {i}: Processing exception: {str(e)}")
        
        return ProcessingResult(
            success=len(all_errors) == 0,
            data=processed_records,
            errors=all_errors,
            warnings=all_warnings,
            processing_time=(datetime.now() - start_time).total_seconds(),
            records_processed=len(records)
        )
    
    def load_from_csv(self, file_path: str, encoding: str = 'utf-8') -> ProcessingResult:
        """
        Load data from CSV file
        
        Args:
            file_path: Path to CSV file
            encoding: File encoding
            
        Returns:
            ProcessingResult with loaded data
        """
        start_time = datetime.now()
        
        try:
            records = []
            with open(file_path, 'r', encoding=encoding) as file:
                reader = csv.DictReader(file)
                records = list(reader)
            
            return ProcessingResult(
                success=True,
                data=records,
                processing_time=(datetime.now() - start_time).total_seconds(),
                records_processed=len(records)
            )
        
        except Exception as e:
            return ProcessingResult(
                success=False,
                errors=[f"CSV loading error: {str(e)}"],
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def save_to_csv(self, data: List[Dict[str, Any]], file_path: str, 
                   encoding: str = 'utf-8') -> ProcessingResult:
        """
        Save data to CSV file
        
        Args:
            data: List of data records
            file_path: Output file path
            encoding: File encoding
            
        Returns:
            ProcessingResult with save status
        """
        start_time = datetime.now()
        
        try:
            if not data:
                return ProcessingResult(
                    success=False,
                    errors=["No data to save"],
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
            
            fieldnames = data[0].keys()
            
            with open(file_path, 'w', newline='', encoding=encoding) as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            
            return ProcessingResult(
                success=True,
                processing_time=(datetime.now() - start_time).total_seconds(),
                records_processed=len(data)
            )
        
        except Exception as e:
            return ProcessingResult(
                success=False,
                errors=[f"CSV saving error: {str(e)}"],
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def generate_report(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a data processing report
        
        Args:
            data: Processed data records
            
        Returns:
            Dictionary containing report statistics
        """
        if not data:
            return {"error": "No data to analyze"}
        
        report = {
            "total_records": len(data),
            "fields": list(data[0].keys()) if data else [],
            "field_statistics": {},
            "data_quality": {},
            "generated_at": datetime.now().isoformat()
        }
        
        # Analyze each field
        for field in report["fields"]:
            field_data = [record.get(field) for record in data if record.get(field) is not None]
            
            field_stats = {
                "total_values": len(field_data),
                "null_count": len(data) - len(field_data),
                "unique_count": len(set(str(val) for val in field_data)),
                "data_types": list(set(type(val).__name__ for val in field_data))
            }
            
            # Add numeric statistics if applicable
            numeric_data = [val for val in field_data if isinstance(val, (int, float))]
            if numeric_data:
                field_stats.update({
                    "mean": np.mean(numeric_data),
                    "median": np.median(numeric_data),
                    "std": np.std(numeric_data),
                    "min": np.min(numeric_data),
                    "max": np.max(numeric_data)
                })
            
            report["field_statistics"][field] = field_stats
        
        # Calculate data quality metrics
        total_fields = len(report["fields"]) * len(data)
        total_nulls = sum(stats["null_count"] for stats in report["field_statistics"].values())
        
        report["data_quality"] = {
            "completeness": (total_fields - total_nulls) / total_fields if total_fields > 0 else 0,
            "total_null_values": total_nulls,
            "fields_with_nulls": sum(1 for stats in report["field_statistics"].values() if stats["null_count"] > 0)
        }
        
        return report
