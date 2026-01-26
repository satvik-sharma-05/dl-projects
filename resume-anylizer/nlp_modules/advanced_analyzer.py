"""
Advanced NLP analyzer for resume analysis
"""

import spacy
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import re
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
import os

class AdvancedResumeAnalyzer:
    def __init__(self, model_path='models/field_classifier.pkl'):
        """Initialize the advanced analyzer"""
        self.nlp = spacy.load('en_core_web_sm')
        self.stop_words = set(stopwords.words('english'))
        
        # Load or create models
        self.model_path = model_path
        self.vectorizer_path = 'models/tfidf_vectorizer.pkl'
        
        if os.path.exists(model_path) and os.path.exists(self.vectorizer_path):
            self.classifier = joblib.load(model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
        else:
            self.classifier = None
            self.vectorizer = None
            self._train_default_model()
    
    def _train_default_model(self):
        """Train a default model if no pre-trained model exists"""
        print("Training default classification model...")
        
        # Create synthetic training data
        training_data = self._create_synthetic_data()
        
        # Prepare features
        texts = training_data['text']
        labels = training_data['label']
        
        # Vectorize text
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        
        X = self.vectorizer.fit_transform(texts)
        
        # Train classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Save models
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.classifier, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)
        
        print(f"Model trained with accuracy: {self.classifier.score(X_test, y_test):.2%}")
    
    def _create_synthetic_data(self):
        """Create synthetic training data for resume classification"""
        data = {
            'text': [],
            'label': []
        }
        
        # Data Science resumes
        ds_resumes = [
            "Experienced Data Scientist with 5 years in machine learning, deep learning, and NLP. "
            "Proficient in Python, TensorFlow, PyTorch, and scikit-learn. "
            "Strong background in statistical analysis and data visualization.",
            
            "Machine Learning Engineer specializing in computer vision and natural language processing. "
            "Experience with ML pipelines, model deployment, and MLOps. "
            "Skills: Python, Docker, Kubernetes, AWS, MLflow.",
            
            "Data Analyst with expertise in SQL, Excel, Tableau, and statistical analysis. "
            "Experience in business intelligence and data-driven decision making."
        ]
        
        # Web Development resumes
        web_resumes = [
            "Full Stack Developer with 4 years experience in React, Node.js, and MongoDB. "
            "Proficient in JavaScript, TypeScript, and modern frontend frameworks. "
            "Experience with REST APIs, microservices, and cloud deployment.",
            
            "Frontend Developer specializing in React and Vue.js. "
            "Strong skills in CSS, HTML5, and responsive design. "
            "Experience with state management (Redux, Vuex) and testing frameworks.",
            
            "Backend Developer with expertise in Python Django and FastAPI. "
            "Experience with PostgreSQL, Redis, and message queues. "
            "Knowledge of system design and scalability."
        ]
        
        # Add data
        for resume in ds_resumes:
            data['text'].append(resume)
            data['label'].append('Data Science')
        
        for resume in web_resumes:
            data['text'].append(resume)
            data['label'].append('Web Development')
        
        # Add more categories as needed
        
        return data
    
    def predict_field(self, resume_text):
        """Predict the field/category of a resume"""
        if not resume_text or len(resume_text.strip()) < 50:
            return {
                'field': 'Unknown',
                'confidence': 0.0,
                'probabilities': {}
            }
        
        # Preprocess text
        processed_text = self._preprocess_text(resume_text)
        
        # Vectorize
        text_vectorized = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.classifier.predict(text_vectorized)[0]
        probabilities = self.classifier.predict_proba(text_vectorized)[0]
        
        # Get class probabilities
        prob_dict = dict(zip(self.classifier.classes_, probabilities))
        
        # Get top 3 predictions
        top_predictions = sorted(
            [(field, prob) for field, prob in prob_dict.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Extract skills mentioned
        skills = self.extract_skills(resume_text)
        
        # Determine missing skills for top field
        missing_skills = self._get_missing_skills(prediction, skills)
        
        return {
            'field': prediction,
            'confidence': float(prob_dict[prediction]),
            'probabilities': prob_dict,
            'top_predictions': top_predictions,
            'skills_found': skills,
            'missing_skills': missing_skills
        }
    
    def extract_skills(self, text):
        """Extract skills from resume text"""
        # Load skills ontology
        skills_ontology = self._load_skills_ontology()
        
        # Convert to lowercase for matching
        text_lower = text.lower()
        
        found_skills = []
        
        # Check for each skill category
        for category, skills in skills_ontology.items():
            for skill in skills:
                # Simple pattern matching
                pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    found_skills.append(skill)
        
        # Remove duplicates and return
        return list(set(found_skills))[:20]  # Return top 20 skills
    
    def _load_skills_ontology(self):
        """Load skills ontology from file or create default"""
        ontology_path = 'data/skills_ontology.json'
        
        if os.path.exists(ontology_path):
            with open(ontology_path, 'r') as f:
                return json.load(f)
        
        # Default skills ontology
        return {
            "programming": [
                "Python", "Java", "JavaScript", "C++", "C#", "Ruby", "Go", "Rust",
                "TypeScript", "PHP", "Swift", "Kotlin", "Scala", "R", "MATLAB"
            ],
            "databases": [
                "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Cassandra",
                "Oracle", "SQLite", "DynamoDB", "Elasticsearch"
            ],
            "frameworks": [
                "React", "Angular", "Vue.js", "Django", "Flask", "Spring",
                "Express.js", "Ruby on Rails", "TensorFlow", "PyTorch"
            ],
            "cloud": [
                "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes",
                "Terraform", "Ansible", "Jenkins", "GitLab CI", "GitHub Actions"
            ],
            "data_science": [
                "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
                "Data Analysis", "Statistics", "Data Visualization",
                "Big Data", "Spark", "Hadoop"
            ],
            "soft_skills": [
                "Communication", "Leadership", "Teamwork", "Problem Solving",
                "Critical Thinking", "Time Management", "Adaptability"
            ]
        }
    
    def _get_missing_skills(self, field, current_skills):
        """Determine missing skills for a given field"""
        field_requirements = {
            "Data Science": [
                "Python", "Machine Learning", "Statistics", "SQL",
                "Data Visualization", "Deep Learning", "NLP"
            ],
            "Web Development": [
                "JavaScript", "HTML", "CSS", "React", "Node.js",
                "Python", "SQL", "Git"
            ],
            "Mobile Development": [
                "Java", "Kotlin", "Swift", "React Native", "Flutter",
                "Android Studio", "Xcode"
            ],
            "DevOps": [
                "Docker", "Kubernetes", "AWS", "Linux", "CI/CD",
                "Python", "Bash", "Terraform"
            ]
        }
        
        if field not in field_requirements:
            return []
        
        required = field_requirements[field]
        current_lower = [s.lower() for s in current_skills]
        
        missing = []
        for skill in required:
            if skill.lower() not in current_lower:
                missing.append(skill)
        
        return missing[:5]  # Return top 5 missing skills
    
    def analyze_sentiment(self, text):
        """Analyze the sentiment/tone of the resume"""
        blob = TextBlob(text)
        
        # Custom analysis for resume tone
        professional_keywords = [
            'achieved', 'implemented', 'developed', 'managed',
            'led', 'increased', 'reduced', 'optimized'
        ]
        
        action_count = sum(1 for word in professional_keywords 
                          if word in text.lower())
        
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'action_word_count': action_count,
            'tone': self._determine_tone(blob.sentiment.polarity, action_count)
        }
    
    def _determine_tone(self, polarity, action_count):
        """Determine the overall tone of the resume"""
        if polarity > 0.1 and action_count > 5:
            return "Confident and Achievement-oriented"
        elif polarity > 0:
            return "Positive and Professional"
        elif action_count > 3:
            return "Action-oriented"
        else:
            return "Neutral"
    
    def extract_experience(self, text):
        """Extract experience information"""
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of)?\s*experience',
            r'experience\s*(?:of)?\s*(\d+)\+?\s*years?',
            r'(\d+)\s*yr',
            r'(\d+)\s*yrs'
        ]
        
        years = 0
        for pattern in experience_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                try:
                    years = max(years, int(matches[0]))
                except:
                    continue
        
        # Extract job roles
        roles = self._extract_job_roles(text)
        
        return {
            'total_years': years,
            'roles': roles,
            'level': self._determine_experience_level(years)
        }
    
    def _extract_job_roles(self, text):
        """Extract job roles from text"""
        common_roles = [
            'software engineer', 'data scientist', 'web developer',
            'machine learning engineer', 'devops engineer', 'data analyst',
            'product manager', 'frontend developer', 'backend developer',
            'full stack developer', 'mobile developer', 'cloud engineer'
        ]
        
        found_roles = []
        for role in common_roles:
            if role in text.lower():
                found_roles.append(role.title())
        
        return found_roles
    
    def _determine_experience_level(self, years):
        """Determine experience level based on years"""
        if years == 0:
            return "Fresher"
        elif years < 3:
            return "Junior"
        elif years < 6:
            return "Mid-level"
        elif years < 10:
            return "Senior"
        else:
            return "Expert/Lead"
    
    def calculate_readability(self, text):
        """Calculate readability scores"""
        # Simple readability metric
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        if len(sentences) == 0:
            return 0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Flesch Reading Ease approximation
        flesch_score = 206.835 - 1.015 * avg_sentence_length - 84.6 * (avg_word_length / 100)
        
        return {
            'flesch_score': max(0, min(100, flesch_score)),
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'word_count': len(words),
            'sentence_count': len(sentences)
        }
    
    def _preprocess_text(self, text):
        """Preprocess text for ML models"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text