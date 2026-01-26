"""
Text preprocessing utilities for resumes
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import spacy
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import unicodedata

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class TextPreprocessor:
    def __init__(self, use_spacy=True):
        """Initialize text preprocessor"""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        if use_spacy:
            try:
                from nlp_modules.spacy_loader import load_spacy_model
                self.nlp = load_spacy_model("en_core_web_sm")

            except:
                print("spaCy model not found. Using NLTK only.")
                self.nlp = None
        else:
            self.nlp = None
        
        # Resume-specific stopwords
        self.resume_stopwords = {
            'email', 'phone', 'mobile', 'address', 'linkedin',
            'github', 'portfolio', 'resume', 'cv', 'contact',
            'objective', 'summary', 'experience', 'education',
            'skills', 'projects', 'certifications', 'achievements'
        }
    
    def preprocess_resume_text(self, text: str, 
                              remove_stopwords: bool = True,
                              lemmatize: bool = True,
                              remove_punctuation: bool = True,
                              remove_numbers: bool = False,
                              lower_case: bool = True) -> str:
        """Preprocess resume text with multiple options"""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        if lower_case:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', '', text)
        
        # Remove special characters and punctuation
        if remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove numbers
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = [token for token in tokens 
                     if token not in self.stop_words 
                     and token not in self.resume_stopwords]
        
        # Lemmatization
        if lemmatize:
            if self.nlp:
                # Use spaCy for better lemmatization
                doc = self.nlp(' '.join(tokens))
                tokens = [token.lemma_ for token in doc]
            else:
                # Use NLTK lemmatizer
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Remove short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract different sections from resume text"""
        sections = {
            'contact': '',
            'summary': '',
            'experience': '',
            'education': '',
            'skills': '',
            'projects': '',
            'certifications': '',
            'achievements': ''
        }
        
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Detect section headers
            if any(keyword in line_lower for keyword in ['contact', 'phone', 'email', 'address']):
                current_section = 'contact'
            elif any(keyword in line_lower for keyword in ['summary', 'objective', 'profile']):
                current_section = 'summary'
            elif any(keyword in line_lower for keyword in ['experience', 'work history', 'employment']):
                current_section = 'experience'
            elif any(keyword in line_lower for keyword in ['education', 'qualification', 'academic']):
                current_section = 'education'
            elif any(keyword in line_lower for keyword in ['skills', 'technical skills', 'competencies']):
                current_section = 'skills'
            elif any(keyword in line_lower for keyword in ['projects', 'portfolio']):
                current_section = 'projects'
            elif any(keyword in line_lower for keyword in ['certifications', 'certificate']):
                current_section = 'certifications'
            elif any(keyword in line_lower for keyword in ['achievements', 'awards', 'honors']):
                current_section = 'achievements'
            elif line.strip() and current_section:
                # Add line to current section
                sections[current_section] += line + '\n'
        
        # Clean sections
        for key in sections:
            sections[key] = sections[key].strip()
        
        return sections
    
    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        if not self.nlp:
            return {}
        
        doc = self.nlp(text)
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],  # Geographical locations
            'DATE': [],
            'NORP': [],  # Nationalities/religious/political groups
            'PRODUCT': []
        }
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def extract_dates(self, text: str) -> List[str]:
        """Extract dates from resume text"""
        date_patterns = [
            r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b',
            r'\b\d{1,2}[/-]\d{4}\b',
            r'\b\d{4}\s*[-–]\s*(?:present|current|\d{4})\b',
            r'\b(?:present|current)\b'
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        return list(set(dates))
    
    def calculate_readability_score(self, text: str) -> Dict[str, float]:
        """Calculate readability scores"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        if len(sentences) == 0 or len(words) == 0:
            return {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'gunning_fog': 0,
                'avg_sentence_length': 0,
                'avg_word_length': 0
            }
        
        # Count syllables (approximate)
        def count_syllables(word):
            word = word.lower()
            count = 0
            vowels = "aeiouy"
            if word[0] in vowels:
                count += 1
            for index in range(1, len(word)):
                if word[index] in vowels and word[index-1] not in vowels:
                    count += 1
            if word.endswith("e"):
                count -= 1
            if count == 0:
                count += 1
            return count
        
        total_syllables = sum(count_syllables(word) for word in words)
        
        # Calculate metrics
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Flesch Reading Ease
        flesch = 206.835 - 1.015 * avg_sentence_length - 84.6 * (total_syllables / len(words))
        flesch = max(0, min(100, flesch))
        
        # Flesch-Kincaid Grade Level
        fk_grade = 0.39 * avg_sentence_length + 11.8 * (total_syllables / len(words)) - 15.59
        
        # Gunning Fog Index
        complex_words = sum(1 for word in words if count_syllables(word) >= 3)
        fog_index = 0.4 * (avg_sentence_length + 100 * (complex_words / len(words)))
        
        return {
            'flesch_reading_ease': round(flesch, 2),
            'flesch_kincaid_grade': round(fk_grade, 2),
            'gunning_fog': round(fog_index, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'avg_word_length': round(avg_word_length, 2)
        }
    
    def extract_keyphrases(self, text: str, top_n: int = 10) -> List[str]:
        """Extract keyphrases using TF-IDF style approach"""
        # Simple approach: find frequent noun phrases
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        
        # Extract noun chunks
        noun_chunks = list(doc.noun_chunks)
        
        # Count frequencies
        chunk_counts = {}
        for chunk in noun_chunks:
            chunk_text = chunk.text.lower().strip()
            if len(chunk_text) > 3 and len(chunk_text.split()) <= 4:
                if chunk_text in chunk_counts:
                    chunk_counts[chunk_text] += 1
                else:
                    chunk_counts[chunk_text] = 1
        
        # Sort by frequency
        sorted_chunks = sorted(chunk_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [chunk for chunk, _ in sorted_chunks[:top_n]]
    
    def normalize_indian_text(self, text: str) -> str:
        """Normalize Indian English text"""
        # Common Indian English variations
        indian_variations = {
            'fresher': 'entry level',
            'lakh': '100,000',
            'crore': '10,000,000',
            'passout': 'graduate',
            'prepone': 'reschedule earlier',
            'revert back': 'respond',
            'do the needful': 'take necessary action',
            'same to you': 'likewise',
            'out of station': 'out of town',
            'kindly': 'please',
            'puc': 'pre-university course',
            'ssc': 'secondary school certificate',
            'hsc': 'higher secondary certificate'
        }
        
        for indian_term, standard_term in indian_variations.items():
            pattern = r'\b' + re.escape(indian_term) + r'\b'
            text = re.sub(pattern, standard_term, text, flags=re.IGNORECASE)
        
        return text
    
    def detect_language(self, text: str) -> str:
        """Detect language of text (simple approach)"""
        # Count English vs Hindi (Devanagari) characters
        english_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
        devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        
        total_chars = len(text.replace(' ', ''))
        
        if total_chars == 0:
            return 'unknown'
        
        english_ratio = english_chars / total_chars
        devanagari_ratio = devanagari_chars / total_chars
        
        if devanagari_ratio > 0.3:
            return 'hindi'
        elif english_ratio > 0.7:
            return 'english'
        else:
            return 'mixed'
    
    def create_feature_vector(self, text: str) -> Dict[str, Any]:
        """Create feature vector from resume text"""
        features = {}
        
        # Basic text features
        features['word_count'] = len(word_tokenize(text))
        features['sentence_count'] = len(sent_tokenize(text))
        features['avg_word_length'] = np.mean([len(word) for word in word_tokenize(text)])
        
        # Readability features
        readability = self.calculate_readability_score(text)
        features.update(readability)
        
        # Section features
        sections = self.extract_sections(text)
        features['has_summary'] = 1 if sections['summary'] else 0
        features['has_objective'] = 1 if 'objective' in text.lower() else 0
        features['has_skills'] = 1 if sections['skills'] else 0
        features['has_projects'] = 1 if sections['projects'] else 0
        
        # Keyword features
        keywords = ['python', 'java', 'machine learning', 'sql', 'aws']
        for keyword in keywords:
            features[f'has_{keyword}'] = 1 if keyword in text.lower() else 0
        
        # Action verb count
        action_verbs = ['achieved', 'implemented', 'developed', 'managed', 'led']
        features['action_verb_count'] = sum(1 for verb in action_verbs if verb in text.lower())
        
        # Quantifiable results
        quant_patterns = [
            r'increased by \d+%',
            r'reduced by \d+%',
            r'\$\d+',  # Dollar amounts
            r'\d+\s*(?:lakh|crore)',  # Indian amounts
        ]
        
        features['quantifiable_results'] = sum(
            1 for pattern in quant_patterns 
            if re.search(pattern, text, re.IGNORECASE)
        )
        
        return features