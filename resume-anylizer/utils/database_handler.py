"""
Database handler for resume analyzer
Gracefully degrades if database is unavailable
"""

import pymysql
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional
import logging
from io import BytesIO

logging.basicConfig(level=logging.INFO)


class DatabaseHandler:
    def __init__(self, config: Optional[Dict] = None):
        """Initialize database handler (DB is optional)"""
        self.config = config or {
            'host': 'localhost',
            'user': 'root',
            'password': 'password',
            'database': 'resume_analyzer',
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor
        }

        self.connection = None
        self.enabled = False
        self.logger = logging.getLogger(__name__)

        try:
            self._initialize_database()
            self.enabled = True
        except Exception as e:
            self.logger.warning(f"⚠️ Database disabled: {e}")
            self.connection = None
            self.enabled = False

    # ------------------------------------------------------------------
    # Core connection handling
    # ------------------------------------------------------------------

    def connect(self):
        if not self.enabled:
            return

        if self.connection is None or not self.connection.open:
            self.connection = pymysql.connect(**self.config)

    def disconnect(self):
        if self.connection and self.connection.open:
            self.connection.close()
            self.connection = None

    def _initialize_database(self):
        self.connect()
        self._create_tables()
        self.logger.info("Database initialized successfully")

    # ------------------------------------------------------------------
    # Table creation
    # ------------------------------------------------------------------

    def _create_tables(self):
        with self.connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    name VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS resumes (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    file_name VARCHAR(500),
                    file_path VARCHAR(500),
                    parsed_data JSON,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    resume_id INT,
                    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    predicted_field VARCHAR(100),
                    confidence_score FLOAT,
                    skills JSON,
                    ats_score INT
                )
            """)

            self.connection.commit()

    # ------------------------------------------------------------------
    # Insert operations (SAFE)
    # ------------------------------------------------------------------

    def insert_analysis(self, analysis_data: Dict):
        if not self.enabled:
            return

        self.connect()

        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO analysis_results
                (resume_id, predicted_field, confidence_score, skills, ats_score)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                analysis_data.get('resume_id', 0),
                analysis_data.get('predicted_field', 'Unknown'),
                analysis_data.get('confidence_score', 0.0),
                json.dumps(analysis_data.get('skills', [])),
                analysis_data.get('ats_score', 0)
            ))

            self.connection.commit()

    # ------------------------------------------------------------------
    # Read operations (SAFE)
    # ------------------------------------------------------------------

    def get_all_analyses(self, limit: int = 100) -> pd.DataFrame:
        if not self.enabled:
            return pd.DataFrame()

        self.connect()

        query = """
            SELECT *
            FROM analysis_results
            ORDER BY analysis_date DESC
            LIMIT %s
        """
        return pd.read_sql(query, self.connection, params=(limit,))

    def get_field_statistics(self) -> pd.DataFrame:
        if not self.enabled:
            return pd.DataFrame()

        self.connect()

        query = """
            SELECT predicted_field, COUNT(*) AS count
            FROM analysis_results
            GROUP BY predicted_field
        """
        return pd.read_sql(query, self.connection)

    # ------------------------------------------------------------------
    # Export utilities
    # ------------------------------------------------------------------

    def export_data(self, format: str = 'csv') -> bytes:
        df = self.get_all_analyses(limit=1000)

        if df.empty:
            return b""

        if format == 'csv':
            return df.to_csv(index=False).encode("utf-8")

        if format == 'json':
            return df.to_json(orient="records").encode("utf-8")

        if format == 'excel':
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            return output.getvalue()

        raise ValueError("Unsupported export format")

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        if self.enabled:
            self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
