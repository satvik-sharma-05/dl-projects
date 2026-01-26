"""
Training script for resume field classifier
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
import os
from datetime import datetime
import logging
from utils.logger import get_logger

# Setup logger
logger = get_logger(__name__)

class FieldClassifierTrainer:
    def __init__(self, data_path='data/resume_dataset.csv'):
        """Initialize classifier trainer"""
        self.data_path = data_path
        self.vectorizer = None
        self.classifier = None
        self.classes = None
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
    
    def load_data(self):
        """Load and prepare training data"""
        logger.info(f"Loading data from {self.data_path}")
        
        if not os.path.exists(self.data_path):
            logger.warning("Training data not found. Creating synthetic data...")
            return self._create_synthetic_data()
        
        try:
            df = pd.read_csv(self.data_path)
            
            # Check required columns
            required_columns = ['text', 'field']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing columns in dataset: {missing_columns}")
                return self._create_synthetic_data()
            
            # Clean data
            df = df.dropna(subset=['text', 'field'])
            df = df[df['text'].str.len() > 50]  # Remove very short texts
            
            logger.info(f"Loaded {len(df)} samples")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self):
        """Create synthetic training data"""
        logger.info("Creating synthetic training data")
        
        synthetic_data = {
            'text': [],
            'field': []
        }
        
        # Data Science resumes
        ds_samples = [
            "Experienced Data Scientist with expertise in machine learning, "
            "deep learning, and natural language processing. Proficient in "
            "Python, TensorFlow, PyTorch, and scikit-learn.",
            
            "Machine Learning Engineer with 5 years experience in developing "
            "and deploying ML models. Strong background in computer vision "
            "and model optimization.",
            
            "Data Analyst skilled in SQL, Excel, Tableau, and statistical "
            "analysis. Experience in business intelligence and data-driven "
            "decision making."
        ]
        
        # Web Development resumes
        web_samples = [
            "Full Stack Developer with expertise in React, Node.js, and MongoDB. "
            "Proficient in JavaScript, TypeScript, and modern web development.",
            
            "Frontend Developer specializing in React and Vue.js. Experience "
            "with responsive design, state management, and modern CSS frameworks.",
            
            "Backend Developer with strong skills in Python Django, FastAPI, "
            "and PostgreSQL. Knowledge of microservices architecture and API design."
        ]
        
        # Mobile Development resumes
        mobile_samples = [
            "Android Developer with 4 years experience in Kotlin and Java. "
            "Proficient in Android SDK, Jetpack components, and material design.",
            
            "iOS Developer skilled in Swift and Objective-C. Experience with "
            "UIKit, Core Data, and Apple's design principles."
        ]
        
        # Add samples
        for sample in ds_samples:
            synthetic_data['text'].append(sample)
            synthetic_data['field'].append('Data Science')
        
        for sample in web_samples:
            synthetic_data['text'].append(sample)
            synthetic_data['field'].append('Web Development')
        
        for sample in mobile_samples:
            synthetic_data['text'].append(sample)
            synthetic_data['field'].append('Mobile Development')
        
        df = pd.DataFrame(synthetic_data)
        logger.info(f"Created {len(df)} synthetic samples")
        
        return df
    
    def preprocess_text(self, texts):
        """Preprocess text data"""
        # Simple preprocessing - in production, use more sophisticated methods
        processed_texts = []
        
        for text in texts:
            if isinstance(text, str):
                # Convert to lowercase
                text = text.lower()
                # Remove special characters
                text = ' '.join(text.split())  # Remove extra whitespace
                processed_texts.append(text)
            else:
                processed_texts.append('')
        
        return processed_texts
    
    def train(self, test_size=0.2, random_state=42):
        """Train the classifier"""
        logger.info("Starting classifier training...")
        
        # Load data
        df = self.load_data()
        
        if len(df) < 10:
            logger.error("Insufficient training data")
            return False
        
        # Preprocess text
        logger.info("Preprocessing text data...")
        df['processed_text'] = self.preprocess_text(df['text'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], df['field'], 
            test_size=test_size, random_state=random_state,
            stratify=df['field']
        )
        
        # Vectorize text
        logger.info("Vectorizing text data...")
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train classifier
        logger.info("Training classifier...")
        
        # Try multiple classifiers
        classifiers = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=random_state
            ),
            'SVM': SVC(
                probability=True,
                random_state=random_state,
                class_weight='balanced'
            )
        }
        
        best_accuracy = 0
        best_classifier = None
        best_classifier_name = ''
        
        for name, clf in classifiers.items():
            logger.info(f"Training {name}...")
            clf.fit(X_train_vec, y_train)
            
            # Evaluate
            y_pred = clf.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"{name} Accuracy: {accuracy:.2%}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_classifier = clf
                best_classifier_name = name
        
        # Set best classifier
        self.classifier = best_classifier
        self.classes = self.classifier.classes_
        
        logger.info(f"Selected {best_classifier_name} with accuracy: {best_accuracy:.2%}")
        
        # Detailed evaluation
        logger.info("\nClassification Report:")
        y_pred = self.classifier.predict(X_test_vec)
        report = classification_report(y_test, y_pred, target_names=self.classes)
        logger.info(report)
        
        # Save models
        self.save_models()
        
        return True
    
    def save_models(self):
        """Save trained models"""
        if self.vectorizer is None or self.classifier is None:
            logger.error("Models not trained yet")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save vectorizer
        vectorizer_path = f'models/tfidf_vectorizer_{timestamp}.pkl'
        joblib.dump(self.vectorizer, vectorizer_path)
        
        # Also save as latest
        joblib.dump(self.vectorizer, 'models/tfidf_vectorizer.pkl')
        
        # Save classifier
        classifier_path = f'models/field_classifier_{timestamp}.pkl'
        joblib.dump(self.classifier, classifier_path)
        
        # Also save as latest
        joblib.dump(self.classifier, 'models/field_classifier.pkl')
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'classes': self.classes.tolist(),
            'vectorizer_features': len(self.vectorizer.get_feature_names_out()),
            'model_type': type(self.classifier).__name__
        }
        
        metadata_path = f'models/classifier_metadata_{timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        with open('models/classifier_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Models saved: {vectorizer_path}, {classifier_path}")
    
    def load_models(self, vectorizer_path='models/tfidf_vectorizer.pkl',
                   classifier_path='models/field_classifier.pkl'):
        """Load trained models"""
        try:
            self.vectorizer = joblib.load(vectorizer_path)
            self.classifier = joblib.load(classifier_path)
            
            # Load metadata
            metadata_path = 'models/classifier_metadata.json'
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.classes = np.array(metadata['classes'])
            
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def predict(self, texts):
        """Predict fields for new texts"""
        if self.vectorizer is None or self.classifier is None:
            logger.error("Models not loaded. Please train or load models first.")
            return None
        
        # Preprocess texts
        processed_texts = self.preprocess_text(texts)
        
        # Vectorize
        X_vec = self.vectorizer.transform(processed_texts)
        
        # Predict
        predictions = self.classifier.predict(X_vec)
        probabilities = self.classifier.predict_proba(X_vec)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            result = {
                'text': texts[i][:100] + '...' if len(texts[i]) > 100 else texts[i],
                'predicted_field': pred,
                'confidence': float(max(probs)),
                'probabilities': dict(zip(self.classes, probs))
            }
            results.append(result)
        
        return results
    
    def evaluate_on_new_data(self, new_data_path):
        """Evaluate classifier on new data"""
        logger.info(f"Evaluating on new data: {new_data_path}")
        
        if not os.path.exists(new_data_path):
            logger.error("New data file not found")
            return
        
        # Load new data
        new_df = pd.read_csv(new_data_path)
        
        if 'text' not in new_df.columns or 'field' not in new_df.columns:
            logger.error("New data must have 'text' and 'field' columns")
            return
        
        # Preprocess
        new_df['processed_text'] = self.preprocess_text(new_df['text'])
        
        # Vectorize
        X_new = self.vectorizer.transform(new_df['processed_text'])
        
        # Predict
        predictions = self.classifier.predict(X_new)
        probabilities = self.classifier.predict_proba(X_new)
        
        # Calculate accuracy
        accuracy = accuracy_score(new_df['field'], predictions)
        logger.info(f"Accuracy on new data: {accuracy:.2%}")
        
        # Detailed report
        logger.info("\nDetailed Classification Report:")
        report = classification_report(new_df['field'], predictions)
        logger.info(report)
        
        # Create confusion matrix (optional)
        self._create_confusion_matrix(new_df['field'], predictions)
        
        return accuracy
    
    def _create_confusion_matrix(self, y_true, y_pred):
        """Create and log confusion matrix"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        cm = confusion_matrix(y_true, y_pred, labels=self.classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.classes,
                   yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save plot
        plot_path = 'models/confusion_matrix.png'
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Confusion matrix saved to {plot_path}")
    
    def retrain_if_needed(self, days_threshold=30):
        """Retrain model if it's older than threshold"""
        model_path = 'models/field_classifier.pkl'
        
        if not os.path.exists(model_path):
            logger.info("No existing model found. Training new model...")
            return self.train()
        
        # Check model age
        model_time = os.path.getmtime(model_path)
        model_age_days = (datetime.now().timestamp() - model_time) / (24 * 3600)
        
        if model_age_days > days_threshold:
            logger.info(f"Model is {model_age_days:.1f} days old. Retraining...")
            return self.train()
        else:
            logger.info(f"Model is recent ({model_age_days:.1f} days old). Skipping retraining.")
            return True

def main():
    """Main training function"""
    logger.info("=" * 50)
    logger.info("RESUME FIELD CLASSIFIER TRAINING")
    logger.info("=" * 50)
    
    # Initialize trainer
    trainer = FieldClassifierTrainer()
    
    # Check if retraining is needed
    trainer.retrain_if_needed(days_threshold=7)  # Weekly retraining
    
    # Or train new model
    # trainer.train()
    
    # Test prediction
    test_texts = [
        "I am a Python developer with experience in Django and Flask.",
        "Machine learning engineer with TensorFlow and PyTorch skills.",
        "React developer with 3 years of frontend experience."
    ]
    
    predictions = trainer.predict(test_texts)
    
    logger.info("\nTest Predictions:")
    for pred in predictions:
        logger.info(f"Text: {pred['text']}")
        logger.info(f"Predicted: {pred['predicted_field']} (Confidence: {pred['confidence']:.2%})")
        logger.info("-" * 30)

if __name__ == "__main__":
    main()