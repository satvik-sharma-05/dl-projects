"""
Configuration management for Resume Analyzer
"""

import yaml
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

class Config:
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration"""
        self.config_path = config_path or 'config.yaml'
        self.config = self._load_config()
        self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            'app': {
                'name': 'AI Resume Analyzer',
                'version': '2.0.0',
                'debug': False,
                'port': 8501,
                'host': '0.0.0.0'
            },
            'database': {
                'host': 'localhost',
                'port': 3306,
                'user': 'root',
                'password': 'password',
                'database': 'resume_analyzer',
                'pool_size': 5,
                'pool_recycle': 3600
            },
            'nlp': {
                'spacy_model': 'en_core_web_sm',
                'sentence_transformer': 'all-MiniLM-L6-v2',
                'enable_gpu': False,
                'batch_size': 16,
                'max_text_length': 10000
            },
            'models': {
                'field_classifier': 'models/field_classifier.pkl',
                'skill_extractor': 'models/skill_extractor.pkl',
                'ats_scorer': 'models/ats_scorer.pkl',
                'retrain_interval': 30  # days
            },
            'paths': {
                'upload_folder': 'Uploaded_Resumes',
                'models_folder': 'models',
                'data_folder': 'data',
                'logs_folder': 'logs',
                'temp_folder': 'temp'
            },
            'security': {
                'admin_username': 'admin',
                'admin_password': 'admin123',
                'session_timeout': 3600,
                'max_file_size': 10 * 1024 * 1024,  # 10 MB
                'allowed_extensions': ['.pdf', '.docx', '.txt']
            },
            'indian_context': {
                'enable': True,
                'universities_file': 'data/indian_universities.json',
                'companies_file': 'data/indian_companies.json',
                'skills_file': 'data/indian_skills.json'
            },
            'recommendations': {
                'max_courses': 5,
                'max_skills': 10,
                'confidence_threshold': 0.7,
                'enable_learning_path': True
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/app.log',
                'max_size': 10 * 1024 * 1024,  # 10 MB
                'backup_count': 5
            },
            'api': {
                'enable': False,
                'port': 8000,
                'rate_limit': '100/hour',
                'cors_origins': ['http://localhost:3000']
            }
        }
        
        # Try to load from file
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    # Merge with default config
                    return self._deep_merge(default_config, file_config)
            except Exception as e:
                print(f"Error loading config file: {e}. Using default config.")
        
        return default_config
    
    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config['logging']
        
        # Create logs directory
        os.makedirs(os.path.dirname(log_config['file']), exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format=log_config['format'],
            handlers=[
                logging.FileHandler(log_config['file']),
                logging.StreamHandler()
            ]
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None):
        """Save configuration to file"""
        save_path = path or self.config_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def create_directories(self):
        """Create all necessary directories"""
        paths = self.config['paths']
        
        for key, path in paths.items():
            os.makedirs(path, exist_ok=True)
        
        # Create other directories
        directories = [
            'static/css',
            'static/images',
            'templates',
            'notebooks',
            'training'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def validate(self) -> bool:
        """Validate configuration"""
        try:
            # Check required directories
            paths = self.config['paths']
            for key, path in paths.items():
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
            
            # Check model files
            models = self.config['models']
            for key, path in models.items():
                if key.endswith('_classifier') or key.endswith('_extractor'):
                    if not os.path.exists(path):
                        print(f"Warning: Model file not found: {path}")
            
            # Validate security settings
            if len(self.config['security']['admin_password']) < 8:
                print("Warning: Admin password is too short")
            
            return True
            
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return self.config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using bracket notation"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Set configuration value using bracket notation"""
        self.set(key, value)