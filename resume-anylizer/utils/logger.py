"""
Logging configuration for Resume Analyzer
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
from typing import Optional

def setup_logger(
    name: str = 'resume_analyzer',
    log_level: str = 'INFO',
    log_file: Optional[str] = None,
    max_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup and configure logger
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, logs to console only)
        max_size: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Create rotating file handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class ResumeAnalyzerLogger:
    """Custom logger for Resume Analyzer with additional features"""
    
    def __init__(self, name: str = 'resume_analyzer', config: Optional[dict] = None):
        """Initialize custom logger"""
        self.config = config or {}
        self.logger = setup_logger(
            name=name,
            log_level=self.config.get('log_level', 'INFO'),
            log_file=self.config.get('log_file'),
            max_size=self.config.get('max_log_size', 10 * 1024 * 1024),
            backup_count=self.config.get('log_backup_count', 5)
        )
        
        # Additional attributes
        self.analysis_logs = []
        self.error_count = 0
        self.warning_count = 0
    
    def log_analysis_start(self, resume_name: str, user_id: Optional[str] = None):
        """Log start of resume analysis"""
        message = f"Starting analysis for resume: {resume_name}"
        if user_id:
            message += f" (User: {user_id})"
        
        self.logger.info(message)
        self.analysis_logs.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'analysis_start',
            'resume_name': resume_name,
            'user_id': user_id
        })
    
    def log_analysis_complete(self, resume_name: str, 
                            field: str, 
                            confidence: float,
                            duration: float):
        """Log completion of resume analysis"""
        message = (f"Analysis complete for {resume_name}: "
                  f"Field={field}, Confidence={confidence:.2%}, "
                  f"Duration={duration:.2f}s")
        
        self.logger.info(message)
        self.analysis_logs.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'analysis_complete',
            'resume_name': resume_name,
            'field': field,
            'confidence': confidence,
            'duration': duration
        })
    
    def log_error(self, error_type: str, error_message: str, 
                 context: Optional[dict] = None):
        """Log error with context"""
        message = f"{error_type}: {error_message}"
        if context:
            message += f" | Context: {context}"
        
        self.logger.error(message)
        self.error_count += 1
        
        self.analysis_logs.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'error',
            'error_type': error_type,
            'error_message': error_message,
            'context': context
        })
    
    def log_warning(self, warning_type: str, warning_message: str,
                   context: Optional[dict] = None):
        """Log warning with context"""
        message = f"{warning_type}: {warning_message}"
        if context:
            message += f" | Context: {context}"
        
        self.logger.warning(message)
        self.warning_count += 1
        
        self.analysis_logs.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'warning',
            'warning_type': warning_type,
            'warning_message': warning_message,
            'context': context
        })
    
    def log_performance(self, operation: str, duration: float,
                       details: Optional[dict] = None):
        """Log performance metrics"""
        message = f"Performance: {operation} took {duration:.2f}s"
        if details:
            message += f" | Details: {details}"
        
        self.logger.info(message)
        
        self.analysis_logs.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'performance',
            'operation': operation,
            'duration': duration,
            'details': details
        })
    
    def log_user_action(self, user_id: str, action: str,
                       details: Optional[dict] = None):
        """Log user actions (for analytics)"""
        message = f"User Action: {user_id} - {action}"
        if details:
            message += f" | Details: {details}"
        
        self.logger.info(message)
        
        self.analysis_logs.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'user_action',
            'user_id': user_id,
            'action': action,
            'details': details
        })
    
    def get_analysis_summary(self) -> dict:
        """Get summary of recent analyses"""
        return {
            'total_analyses': len([log for log in self.analysis_logs 
                                  if log['event'] == 'analysis_complete']),
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'recent_logs': self.analysis_logs[-10:] if self.analysis_logs else []
        }
    
    def export_logs(self, format: str = 'json') -> str:
        """Export logs in specified format"""
        import json
        import csv
        from io import StringIO
        
        if format == 'json':
            return json.dumps(self.analysis_logs, indent=2)
        elif format == 'csv':
            if not self.analysis_logs:
                return ''
            
            # Convert to CSV
            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=self.analysis_logs[0].keys())
            writer.writeheader()
            writer.writerows(self.analysis_logs)
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def clear_logs(self):
        """Clear stored logs"""
        self.analysis_logs = []
        self.error_count = 0
        self.warning_count = 0
    
    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying logger"""
        return getattr(self.logger, name)

# Global logger instance
_logger_instance = None

def get_logger(name: str = 'resume_analyzer', config: Optional[dict] = None):
    """Get or create global logger instance"""
    global _logger_instance
    
    if _logger_instance is None:
        _logger_instance = ResumeAnalyzerLogger(name, config)
    
    return _logger_instance