"""
Enhanced semantic matching between resumes and job descriptions with proper skill extraction
FIXED VERSION with correct experience calculation
"""

import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib
from collections import defaultdict
from datetime import datetime

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

class EnhancedSemanticMatcher:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize enhanced semantic matcher"""
        self.model = SentenceTransformer(model_name)
        self.stop_words = set(stopwords.words('english'))
        
        # Comprehensive skill database
        self.technical_skills = self._load_technical_skills()
        
        # Experience patterns
        self.experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of)?\s*experience',
            r'experience\s*(?:of)?\s*(\d+)\+?\s*years?',
            r'(\d+)\s*(?:to\s*)?\d*\s*years?',
            r'(\d+)\s*yr',
            r'(\d+)\s*yrs',
            r'minimum\s*(\d+)\s*years',
            r'at\s*least\s*(\d+)\s*years'
        ]
    
    def _load_technical_skills(self):
        """Load comprehensive technical skills database"""
        return {
            'programming_languages': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 
                'go', 'rust', 'php', 'swift', 'kotlin', 'scala', 'r', 'matlab'
            ],
            'frontend': [
                'react', 'angular', 'vue', 'next.js', 'nuxt.js', 'svelte', 
                'html', 'css', 'sass', 'less', 'tailwind css', 'bootstrap',
                'redux', 'context api', 'vuex', 'zustand', 'jquery'
            ],
            'backend': [
                'node.js', 'express', 'django', 'flask', 'fastapi', 'spring',
                'spring boot', 'laravel', 'ruby on rails', 'asp.net', 'net core',
                'graphql', 'rest api', 'soap', 'microservices'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
                'sqlite', 'oracle', 'sql server', 'cassandra', 'dynamodb',
                'firebase', 'supabase', 'prisma'
            ],
            'devops_cloud': [
                'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'terraform',
                'ansible', 'jenkins', 'gitlab ci', 'github actions', 'circleci',
                'prometheus', 'grafana', 'nginx', 'apache', 'linux', 'bash'
            ],
            'data_science_ai': [
                'machine learning', 'deep learning', 'tensorflow', 'pytorch',
                'keras', 'scikit-learn', 'pandas', 'numpy', 'opencv',
                'nlp', 'computer vision', 'data analysis', 'tableau',
                'power bi', 'matplotlib', 'seaborn', 'plotly'
            ],
            'mobile': [
                'react native', 'flutter', 'ios', 'android', 'swift',
                'kotlin', 'objective-c', 'xcode', 'android studio'
            ],
            'testing': [
                'jest', 'cypress', 'selenium', 'pytest', 'junit',
                'unit testing', 'integration testing', 'end-to-end testing',
                'tdd', 'bdd', 'test automation'
            ],
            'tools_methodologies': [
                'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence',
                'agile', 'scrum', 'kanban', 'devops', 'ci/cd', 'tdd'
            ],
            'soft_skills': [
                'communication', 'teamwork', 'leadership', 'problem solving',
                'time management', 'adaptability', 'critical thinking',
                'creativity', 'attention to detail', 'collaboration'
            ]
        }
    
    def match_resume_jd(self, resume_text, jd_text):
        """Enhanced matching with proper skill extraction"""
        # Extract detailed skills
        resume_skills = self._extract_detailed_skills(resume_text)
        jd_skills = self._extract_detailed_skills(jd_text)
        
        # Calculate skill-based match
        skill_match_result = self._calculate_skill_match(resume_skills, jd_skills)
        
        # Calculate semantic similarity
        semantic_score = self._calculate_semantic_similarity(resume_text, jd_text)
        
        # Calculate experience match
        experience_match = self._match_experience_enhanced(resume_text, jd_text)
        
        # Calculate overall match
        overall_score = self._calculate_overall_score(
            skill_match_result, semantic_score, experience_match
        )
        
        # Generate detailed insights
        insights = self._generate_detailed_insights(
            resume_skills, jd_skills, skill_match_result, experience_match
        )
        
        return {
            'match_score': round(overall_score * 100, 1),
            'skill_match_score': round(skill_match_result['overall_score'] * 100, 1),
            'semantic_score': round(semantic_score * 100, 1),
            'experience_match_score': round(experience_match['match_score'] * 100, 1),
            'strengths': skill_match_result['matched_skills'][:15],
            'missing_skills': skill_match_result['missing_skills'][:15],
            'experience_analysis': experience_match,
            'category_analysis': skill_match_result['category_scores'],
            'recommendations': insights['recommendations'],
            'priority_skills': insights['priority_skills']
        }
    
    def _extract_detailed_skills(self, text):
        """Extract skills with categories"""
        text_lower = text.lower()
        skills_by_category = defaultdict(list)
        
        # Extract skills from each category
        for category, skill_list in self.technical_skills.items():
            for skill in skill_list:
                # Create regex pattern for skill matching
                skill_pattern = r'\b' + re.escape(skill) + r'\b'
                
                # Check for skill variations
                if category == 'programming_languages':
                    # Also check for language mentions like "Python developer"
                    pattern = rf'\b{re.escape(skill)}\b'
                elif ' ' in skill:
                    # Multi-word skills
                    pattern = rf'\b{re.escape(skill)}\b'
                else:
                    # Single word skills
                    pattern = rf'\b{re.escape(skill)}(?:\s*(?:js|\.js|\.jsx|\.ts|\.tsx|\.py|\.java|\.cpp))?\b'
                
                if re.search(pattern, text_lower, re.IGNORECASE):
                    skills_by_category[category].append(skill)
        
        return dict(skills_by_category)
    
    def _calculate_skill_match(self, resume_skills, jd_skills):
        """Calculate skill match with category breakdown"""
        if not jd_skills:
            return {
                'overall_score': 0,
                'matched_skills': [],
                'missing_skills': [],
                'category_scores': {}
            }
        
        total_jd_skills = 0
        matched_skills = []
        missing_skills = []
        category_scores = {}
        
        # Calculate match for each category
        for category in self.technical_skills.keys():
            jd_skills_in_category = set(jd_skills.get(category, []))
            resume_skills_in_category = set(resume_skills.get(category, []))
            
            total_jd_skills += len(jd_skills_in_category)
            
            if jd_skills_in_category:
                # Skills matched in this category
                matched_in_category = jd_skills_in_category.intersection(resume_skills_in_category)
                missing_in_category = jd_skills_in_category - resume_skills_in_category
                
                matched_skills.extend(matched_in_category)
                missing_skills.extend(missing_in_category)
                
                # Category score
                category_score = len(matched_in_category) / len(jd_skills_in_category) if jd_skills_in_category else 0
                category_scores[category] = round(category_score * 100, 2)
        
        # Overall score
        overall_score = len(matched_skills) / total_jd_skills if total_jd_skills > 0 else 0
        
        return {
            'overall_score': overall_score,
            'matched_skills': sorted(matched_skills),
            'missing_skills': sorted(missing_skills),
            'category_scores': category_scores
        }
    
    def _calculate_semantic_similarity(self, text1, text2):
        """Calculate semantic similarity using sentence transformers"""
        try:
            embedding1 = self.model.encode(text1, convert_to_tensor=True)
            embedding2 = self.model.encode(text2, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(embedding1, embedding2)
            return similarity.item()
        except:
            return 0.0
    
    def _match_experience_enhanced(self, resume_text, jd_text):
        """Enhanced experience matching"""
        # Extract experience from JD
        jd_experience = self._extract_jd_experience(jd_text)
        
        # Extract experience from resume
        resume_experience = self._extract_resume_experience(resume_text)
        
        # Calculate match score
        if jd_experience == 0:
            match_score = 1.0
        else:
            match_score = min(1.0, resume_experience / jd_experience)
        
        return {
            'jd_required': jd_experience,
            'resume_has': resume_experience,
            'match_score': match_score,
            'meets_requirement': resume_experience >= jd_experience,
            'gap_years': max(0, jd_experience - resume_experience)
        }
    
    def _extract_jd_experience(self, text):
        """Extract years of experience from job description"""
        text_lower = text.lower()
        
        for pattern in self.experience_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    years = []
                    for match in matches:
                        if isinstance(match, tuple):
                            for m in match:
                                if m.isdigit():
                                    years.append(int(m))
                        elif isinstance(match, str) and match.isdigit():
                            years.append(int(match))
                    
                    if years:
                        # For JD, take the maximum requirement
                        return max(years)
                except:
                    continue
        
        # Look for experience in roles for JD
        experience_keywords = [
            ('fresher', 0),
            ('entry level', 0),
            ('junior', 1),
            ('mid level', 3),
            ('senior', 5),
            ('lead', 7),
            ('principal', 8),
            ('architect', 10)
        ]
        
        for keyword, years in experience_keywords:
            if keyword in text_lower:
                return years
        
        return 0
    
    def _extract_resume_experience(self, text):
        """Extract actual work experience from resume (not education)"""
        text_lower = text.lower()
        
        # First, check if this is clearly a student/fresher
        student_keywords = ['student', 'fresher', 'bachelor', 'b.tech', 'undergraduate', 'college']
        if any(keyword in text_lower for keyword in student_keywords):
            # Check if they have internship experience
            if 'intern' in text_lower or 'internship' in text_lower:
                return self._calculate_internship_experience(text_lower)
            # Student with projects but no internship
            elif 'project' in text_lower or 'github' in text_lower:
                return 0.1  # Some project experience
            else:
                return 0.0  # Complete fresher
        
        # Try to find explicit experience mentions
        for pattern in self.experience_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    years = []
                    for match in matches:
                        if isinstance(match, tuple):
                            for m in match:
                                if m.isdigit():
                                    years.append(int(m))
                        elif isinstance(match, str) and match.isdigit():
                            years.append(int(match))
                    
                    if years:
                        # For resume, take what's mentioned
                        return min(max(years), 30)  # Cap at reasonable
                except:
                    continue
        
        # Calculate from work dates
        return self._calculate_work_experience_from_dates(text_lower)
    
    def _calculate_internship_experience(self, text_lower):
        """Calculate internship experience in years"""
        # Look for internship duration
        patterns = [
            r'(\d+)\s*(?:month|mo|m)\s*(?:intern|internship)',
            r'intern.*?(\d{4}).*?(\d{4})',
            r'(?:may|jun|jul|aug|sep|oct|nov|dec).*?(\d{4}).*?(?:may|jun|jul|aug|sep|oct|nov|dec).*?(\d{4})'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                if 'month' in pattern or 'mo' in pattern:
                    for match in matches:
                        if isinstance(match, tuple):
                            for m in match:
                                if m.isdigit():
                                    months = int(m)
                                    return round(months / 12.0, 1)
                        elif isinstance(match, str) and match.isdigit():
                            months = int(match)
                            return round(months / 12.0, 1)
                else:
                    # Found date range, count as at least 1 month
                    return 0.1
        
        # Default internship length
        return 0.1  # 1-month internship
    
    def _calculate_work_experience_from_dates(self, text_lower):
        """Calculate work experience from date ranges"""
        date_patterns = [
            r'(\d{4})\s*[-–]\s*(\d{4}|present|current|now)',
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{4})\s*[-–]\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{4})',
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{4})\s*[-–]\s*(present|current|now)'
        ]
        
        total_years = 0
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) >= 2:
                        try:
                            # Try to extract years
                            if match[0].isdigit() and match[1].isdigit():
                                start = int(match[0])
                                end = int(match[1])
                                total_years += (end - start)
                            else:
                                # Count as at least 1 year
                                total_years += 1
                        except:
                            total_years += 1
        
        return min(total_years, 30)  # Cap at 30 years
    
    def _calculate_overall_score(self, skill_match, semantic_score, experience_match):
        """Calculate weighted overall score"""
        weights = {
            'skill_match': 0.5,
            'semantic_score': 0.3,
            'experience_match': 0.2
        }
        
        overall_score = (
            weights['skill_match'] * skill_match['overall_score'] +
            weights['semantic_score'] * semantic_score +
            weights['experience_match'] * experience_match['match_score']
        )
        
        return min(1.0, overall_score)
    
    def _generate_detailed_insights(self, resume_skills, jd_skills, skill_match, experience_match):
        """Generate detailed insights and recommendations"""
        recommendations = []
        priority_skills = []
        
        # Skill-based recommendations
        if skill_match['missing_skills']:
            # Categorize missing skills by importance
            critical_categories = ['programming_languages', 'backend', 'frontend', 'databases']
            
            missing_critical = []
            missing_other = []
            
            for skill in skill_match['missing_skills'][:10]:
                # Find category of missing skill
                skill_category = None
                for category, skills in self.technical_skills.items():
                    if skill in skills:
                        skill_category = category
                        break
                
                if skill_category in critical_categories:
                    missing_critical.append(skill)
                else:
                    missing_other.append(skill)
            
            if missing_critical:
                recommendations.append({
                    'type': 'critical',
                    'message': f"Missing critical skills: {', '.join(missing_critical[:5])}",
                    'action': "Focus on learning these skills or highlighting related experience"
                })
                priority_skills.extend(missing_critical)
            
            if missing_other:
                recommendations.append({
                    'type': 'important',
                    'message': f"Additional skills needed: {', '.join(missing_other[:5])}",
                    'action': "Consider adding these to your skills section"
                })
                priority_skills.extend(missing_other)
        
        # Experience-based recommendations
        if not experience_match['meets_requirement']:
            gap = experience_match['gap_years']
            recommendations.append({
                'type': 'experience',
                'message': f"Experience gap: {gap} year{'s' if gap > 1 else ''}",
                'action': f"Highlight transferable skills and relevant projects to compensate"
            })
        
        # Category analysis
        low_categories = []
        for category, score in skill_match['category_scores'].items():
            if score < 50:  # Less than 50% match
                low_categories.append(category.replace('_', ' ').title())
        
        if low_categories:
            recommendations.append({
                'type': 'category',
                'message': f"Need improvement in: {', '.join(low_categories[:3])}",
                'action': "Focus on developing skills in these areas"
            })
        
        # General recommendations based on score
        overall_percentage = skill_match['overall_score'] * 100
        
        if overall_percentage < 40:
            recommendations.append({
                'type': 'overall',
                'message': "Low skill match - consider if this role aligns with your expertise",
                'action': "Look for roles better matching your current skill set"
            })
        elif overall_percentage < 60:
            recommendations.append({
                'type': 'overall',
                'message': "Moderate match - good foundation but needs improvement",
                'action': "Tailor resume to highlight relevant skills more prominently"
            })
        elif overall_percentage < 80:
            recommendations.append({
                'type': 'overall',
                'message': "Good match - you're a strong candidate",
                'action': "Prepare for technical interviews in your weaker areas"
            })
        else:
            recommendations.append({
                'type': 'overall',
                'message': "Excellent match - highly relevant candidate",
                'action': "Focus on interview preparation and salary negotiation"
            })
        
        return {
            'recommendations': recommendations,
            'priority_skills': priority_skills[:10]
        }