"""
Enhanced resume parser with section-aware parsing
Uses deterministic rules + section isolation for accurate parsing
"""

import re
import json
import os
from datetime import datetime
import spacy
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

class EnhancedResumeParser:
    def __init__(self):
        """Initialize enhanced resume parser"""
        try:
            from nlp_modules.spacy_loader import load_spacy_model
            self.nlp = load_spacy_model()

        except:
            print("⚠️ spaCy model not found. Using fallback parsing only.")
            self.nlp = None
        
        # Patterns for deterministic extraction
        self.patterns = {
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'phone': r'(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}',
            'linkedin': r'linkedin\.com/in/[a-zA-Z0-9\-_]+',
            'github': r'github\.com/[a-zA-Z0-9\-_]+',
        }
        
        # Section headers (uppercase in resumes)
        self.section_headers = [
            'EDUCATION', 'EXPERIENCE', 'WORK EXPERIENCE', 
            'PROJECTS', 'SKILLS', 'CERTIFICATIONS',
            'ACHIEVEMENTS', 'PUBLICATIONS', 'CONTACT',
            'OBJECTIVE', 'SUMMARY', 'PROFILE'
        ]
        
        # Indian context data (for post-processing)
        self.indian_universities = [
            'Thapar Institute of Engineering and Technology',
            'Indian Institute of Technology', 'IIT', 'IIT Bombay',
            'Indian Institute of Management', 'IIM', 'Delhi University',
            'University of Delhi', 'Vellore Institute of Technology',
            'Manipal University', 'Birla Institute of Technology'
        ]
    
    def parse_resume(self, file_path: str) -> Dict[str, Any]:
        """Parse resume from file with section-aware parsing"""
        # Extract text based on file type
        text = self._extract_text_from_file(file_path)
        
        if not text or len(text.strip()) < 50:
            return self._create_empty_result()
        
        # Main parsing pipeline
        result = self._parse_with_sections(text)
        
        # Enhance with additional analysis
        enhanced = self._enhance_parsing(text, result)
        
        return enhanced
    
    def _extract_text_from_file(self, file_path: str) -> str:
        """Extract text from different file formats"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.pdf':
                from pdfminer.high_level import extract_text
                return extract_text(file_path)
            elif file_ext == '.docx':
                import docx
                doc = docx.Document(file_path)
                return '\n'.join([para.text for para in doc.paragraphs])
            elif file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                return ""
        except Exception as e:
            print(f"⚠️ Error extracting text: {e}")
            return ""
    
    def _parse_with_sections(self, text: str) -> Dict[str, Any]:
        """Parse resume using section-aware approach"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # 1. Extract sections FIRST
        sections = self._extract_sections_deterministic(lines)
        
        # 2. Parse each section independently
        result = {
            'name': self._extract_name_deterministic(lines),
            'email': self._extract_email_deterministic(text),
            'phone': self._extract_phone_deterministic(text),
            'sections': sections,
            'education': self._parse_education_section(sections.get('EDUCATION', '')),
            'skills': self._parse_skills_section(sections.get('SKILLS', '')),
            'projects': self._parse_projects_section(sections.get('PROJECTS', '')),
            'experience': self._parse_experience_section(sections.get('EXPERIENCE', sections.get('WORK EXPERIENCE', ''))),
            'raw_text': text[:2000]
        }
        
        return result
    
    def _extract_sections_deterministic(self, lines: List[str]) -> Dict[str, str]:
        """Extract sections based on headers"""
        sections = {}
        current_section = None
        current_content = []
        
        for line in lines:
            # Check if line is a section header (uppercase, common header)
            is_header = False
            
            # Check exact match for common headers
            if line.upper() in self.section_headers:
                is_header = True
                header_name = line.upper()
            # Check for headers with colons
            elif ':' in line and line.split(':')[0].strip().upper() in self.section_headers:
                is_header = True
                header_name = line.split(':')[0].strip().upper()
            # Check for common header patterns
            elif any(header in line.upper() for header in self.section_headers):
                is_header = True
                # Find which header it is
                for header in self.section_headers:
                    if header in line.upper():
                        header_name = header
                        break
            
            if is_header:
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = header_name
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Save last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _extract_name_deterministic(self, lines: List[str]) -> str:
        """Extract name using deterministic rules (NO spaCy)"""
        if not lines:
            return ""
        
        # Rule 1: First non-empty line that looks like a name
        for i in range(min(5, len(lines))):  # Check first 5 lines
            line = lines[i]
            
            # Name must: have 2-4 words, no digits, not an email/phone
            words = line.split()
            if 2 <= len(words) <= 4:
                has_digits = any(char.isdigit() for char in line)
                has_email = '@' in line
                has_phone = any(char.isdigit() for char in line) and len(line.replace(' ', '').replace('-', '').replace('+', '')) >= 10
                
                if not has_digits and not has_email and not has_phone:
                    # Check if it's not a section header
                    if line.upper() not in self.section_headers:
                        return line
        
        # Rule 2: Look for name pattern in first few lines
        name_patterns = [
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+$',  # First Last
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+$'  # First Middle Last
        ]
        
        for i in range(min(3, len(lines))):
            line = lines[i]
            for pattern in name_patterns:
                if re.match(pattern, line):
                    return line
        
        return ""
    
    def _extract_email_deterministic(self, text: str) -> str:
        """Extract email using regex"""
        match = re.search(self.patterns['email'], text)
        return match.group(0) if match else ""
    
    def _extract_phone_deterministic(self, text: str) -> str:
        """Extract phone using regex"""
        match = re.search(self.patterns['phone'], text)
        return match.group(0) if match else ""
    
    def _parse_education_section(self, section_text: str) -> List[Dict[str, str]]:
        """Parse EDUCATION section specifically"""
        if not section_text:
            return []
        
        education_entries = []
        lines = [line.strip() for line in section_text.split('\n') if line.strip()]
        
        current_entry = {}
        collecting_details = False
        
        for line in lines:
            # Detect institution lines (contains education keywords)
            edu_keywords = ['university', 'college', 'institute', 'school', 'academy']
            if any(keyword in line.lower() for keyword in edu_keywords):
                # Save previous entry
                if current_entry:
                    education_entries.append(current_entry)
                
                # Start new entry
                current_entry = {
                    'institution': line,
                    'details': [],
                    'degree': '',
                    'year': '',
                    'gpa': ''
                }
                collecting_details = True
            elif collecting_details and current_entry:
                current_entry['details'].append(line)
                
                # Extract specific information
                # Degree
                degree_patterns = [
                    r'(bachelor|b\.?tech|b\.?e|be|bsc|ba|b\.?a)',
                    r'(master|m\.?tech|m\.?e|me|msc|ma|m\.?a)',
                    r'(phd|ph\.?d|doctorate)',
                    r'(diploma|certificate)'
                ]
                
                for pattern in degree_patterns:
                    if re.search(pattern, line.lower()):
                        current_entry['degree'] = line
                        break
                
                # GPA
                gpa_match = re.search(r'GPA[:\s]*([\d\.]+)', line, re.IGNORECASE)
                if gpa_match:
                    current_entry['gpa'] = gpa_match.group(1)
                
                # Year
                year_match = re.search(r'(20\d{2}|19\d{2})', line)
                if year_match:
                    current_entry['year'] = year_match.group(1)
        
        # Save last entry
        if current_entry:
            education_entries.append(current_entry)
        
        # Clean up details
        for entry in education_entries:
            if entry['details']:
                entry['details'] = ' | '.join(entry['details'][:3])
            else:
                entry['details'] = ''
        
        return education_entries
    
    def _parse_skills_section(self, section_text: str) -> Dict[str, List[str]]:
        """Parse SKILLS section with categorization"""
        if not section_text:
            return {}
        
        # Categorized skills
        categorized = {
            'Programming': [],
            'Web Development': [],
            'Data Science': [],
            'Tools': [],
            'Frameworks': [],
            'Other': []
        }
        
        # Skill patterns
        skill_patterns = {
            'Programming': ['python', 'java', 'javascript', 'c++', 'c#', 'c', 'r', 'sql'],
            'Web Development': ['html', 'css', 'react', 'angular', 'vue', 'django', 'flask', 'node'],
            'Data Science': ['pandas', 'numpy', 'matplotlib', 'tensorflow', 'pytorch', 'machine learning'],
            'Tools': ['git', 'github', 'docker', 'aws', 'azure', 'jupyter', 'vs code', 'colab'],
            'Frameworks': ['tailwind', 'bootstrap', 'spring', 'express', 'fastapi']
        }
        
        # Parse by lines
        lines = [line.strip() for line in section_text.split('\n') if line.strip()]
        
        for line in lines:
            # Handle "Category: skill1, skill2, ..." format
            if ':' in line:
                category_part, skills_part = line.split(':', 1)
                skills = [s.strip() for s in skills_part.split(',') if s.strip()]
                
                # Check category
                category_found = False
                for cat_name, cat_keywords in skill_patterns.items():
                    if any(keyword in category_part.lower() for keyword in cat_keywords):
                        categorized[cat_name].extend(skills)
                        category_found = True
                        break
                
                if not category_found:
                    categorized['Other'].extend(skills)
            else:
                # Just comma-separated skills
                skills = [s.strip() for s in line.split(',') if s.strip()]
                for skill in skills:
                    skill_lower = skill.lower()
                    
                    skill_categorized = False
                    for cat_name, cat_keywords in skill_patterns.items():
                        if any(keyword in skill_lower for keyword in cat_keywords):
                            categorized[cat_name].append(skill)
                            skill_categorized = True
                            break
                    
                    if not skill_categorized:
                        categorized['Other'].append(skill)
        
        # Remove duplicates and empty categories
        for category in list(categorized.keys()):
            if categorized[category]:
                categorized[category] = list(set(categorized[category]))
            else:
                del categorized[category]
        
        return categorized
    
    def _parse_projects_section(self, section_text: str) -> List[Dict[str, Any]]:
        """Parse PROJECTS section"""
        if not section_text:
            return []
        
        projects = []
        lines = [line.strip() for line in section_text.split('\n') if line.strip()]
        
        current_project = {}
        collecting = False
        
        for line in lines:
            # Project title detection
            # Titles are often short, not bullet points, might have keywords
            is_title = (
                len(line) < 80 and 
                not line.startswith('•') and 
                not line.startswith('-') and
                any(keyword in line.lower() for keyword in ['project', 'app', 'system', 'manager', 'page'])
            )
            
            if is_title:
                # Save previous project
                if current_project:
                    projects.append(current_project)
                
                # Start new project
                current_project = {
                    'title': line,
                    'description': [],
                    'tech_stack': [],
                    'links': {}
                }
                collecting = True
            elif collecting and current_project:
                if line.startswith('•') or line.startswith('-'):
                    # Bullet point
                    bullet = line[1:].strip()
                    current_project['description'].append(bullet)
                    
                    # Extract tech stack
                    tech_keywords = ['html', 'css', 'javascript', 'react', 'python', 
                                   'django', 'flask', 'sql', 'aws', 'docker']
                    for tech in tech_keywords:
                        if tech in bullet.lower():
                            current_project['tech_stack'].append(tech.title())
                    
                    # Extract links
                    if 'github' in bullet.lower():
                        github_match = re.search(r'github\.com/[a-zA-Z0-9\-_/]+', bullet.lower())
                        if github_match:
                            current_project['links']['github'] = f"https://{github_match.group(0)}"
                    
                    if 'demo' in bullet.lower() or 'live' in bullet.lower():
                        url_match = re.search(r'https?://[^\s]+', bullet)
                        if url_match:
                            current_project['links']['demo'] = url_match.group(0)
                else:
                    # Regular line
                    current_project['description'].append(line)
        
        # Save last project
        if current_project:
            projects.append(current_project)
        
        # Clean up
        for project in projects:
            project['description'] = ' '.join(project['description'][:3])  # First 3 points
            project['tech_stack'] = list(set(project['tech_stack']))
        
        return projects
    
    def _parse_experience_section(self, section_text: str) -> List[Dict[str, str]]:
        """Parse EXPERIENCE section"""
        if not section_text:
            return []
        
        experiences = []
        lines = [line.strip() for line in section_text.split('\n') if line.strip()]
        
        current_exp = {}
        collecting = False
        
        for line in lines:
            # Company/position line detection
            # Usually contains company name and may contain dates
            has_company_keywords = any(keyword in line.lower() for keyword in 
                                     ['at', 'inc', 'ltd', 'corporation', 'technologies'])
            has_date_pattern = re.search(r'(20\d{2}|19\d{2})', line)
            
            if has_company_keywords or has_date_pattern:
                # Save previous experience
                if current_exp:
                    experiences.append(current_exp)
                
                # Start new experience
                current_exp = {
                    'position': line,
                    'details': [],
                    'company': '',
                    'duration': ''
                }
                collecting = True
                
                # Try to extract company
                if ' at ' in line.lower():
                    parts = line.split(' at ')
                    if len(parts) > 1:
                        current_exp['company'] = parts[1].strip()
                elif ' - ' in line:
                    parts = line.split(' - ')
                    if len(parts) > 1:
                        current_exp['company'] = parts[0].strip()
            elif collecting and current_exp:
                if line.startswith('•') or line.startswith('-'):
                    current_exp['details'].append(line[1:].strip())
                else:
                    # Check for duration in non-bullet lines
                    duration_match = re.search(r'(\d{4}).*(\d{4}|present|current)', line.lower())
                    if duration_match:
                        current_exp['duration'] = line
        
        # Save last experience
        if current_exp:
            experiences.append(current_exp)
        
        # Clean up
        for exp in experiences:
            exp['details'] = ' | '.join(exp['details'][:3])  # First 3 bullet points
        
        return experiences
    
    def _enhance_parsing(self, text: str, base_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance base parsing with additional analysis"""
        enhanced = base_result.copy()
        
        # Calculate total experience
        enhanced['total_experience'] = self._calculate_experience_years(base_result['experience'])
        
        # Determine experience level
        enhanced['experience_level'] = self._determine_experience_level(
            enhanced['total_experience'],
            len(base_result.get('projects', [])),
            len(base_result.get('experience', []))
        )
        
        # Extract certifications
        enhanced['certifications'] = self._extract_certifications(text)
        
        # Extract portfolio links
        enhanced['portfolio_links'] = self._extract_portfolio_links(text)
        
        # Indian context analysis
        enhanced['indian_context'] = self._analyze_indian_context(base_result)
        
        # Field prediction
        enhanced['predicted_field'] = self._predict_field(base_result)
        
        # Skill count
        total_skills = 0
        for skills in base_result.get('skills', {}).values():
            total_skills += len(skills)
        enhanced['total_skills'] = total_skills
        
        return enhanced
    
    def _calculate_experience_years(self, experiences: List[Dict]) -> float:
        """Calculate total years of WORK experience (not education)"""
        if not experiences:
            return 0.0
    
        total_years = 0.0
    
        for exp in experiences:
            duration = exp.get('duration', '').lower()
        
            # Skip if this is education-related (internships are OK)
            if 'intern' in exp.get('position', '').lower() or 'intern' in exp.get('company', '').lower():
                # Internship - count as partial year
                # Try to extract months
                month_patterns = [
                    r'(\d+)\s*month',
                    r'(\d+)\s*m',
                    r'(\d{4}).*(\d{4})'  # Year range
                ]
            
                months = 0
                for pattern in month_patterns:
                    matches = re.findall(pattern, duration)
                    if matches:
                        if pattern == r'(\d{4}).*(\d{4})':
                            try:
                                start_year = int(matches[0][0])
                                end_year = int(matches[0][1])
                                months = (end_year - start_year) * 12
                            except:
                                continue
                        else:
                            try:
                                months = int(matches[0])
                            except:
                                continue
            
                # Default internship: 2-6 months
                if months == 0:
                    months = 3  # Default 3 months
            
                total_years += months / 12.0
        
            else:
                # Regular job - try to extract years
                year_matches = re.findall(r'20\d{2}|19\d{2}', duration)
                if len(year_matches) >= 2:
                    try:
                        years = abs(int(year_matches[-1]) - int(year_matches[0]))
                        total_years += years
                    except:
                        # If can't parse, assume at least 1 year
                        if duration.strip():
                            total_years += 1.0
                elif duration.strip():
                    # Has duration but no clear years
                    total_years += 1.0
    
        return round(total_years, 1)  # Return with 1 decimal
        return min(total_years, 30)  # Cap at 30 years
    
    def _determine_experience_level(self, experience_years: int, 
                                   project_count: int, exp_count: int) -> str:
        """Determine experience level"""
        if experience_years == 0:
            if project_count >= 3:
                return "Beginner with Projects"
            elif project_count >= 1:
                return "Fresher with Projects"
            else:
                return "Fresher"
        elif experience_years < 2:
            return "Junior"
        elif experience_years < 5:
            return "Mid-Level"
        else:
            return "Senior"
    
    def _extract_certifications(self, text: str) -> List[str]:
        """Extract certifications"""
        cert_keywords = [
            'certified', 'certification', 'aws certified',
            'azure certified', 'google cloud', 'oracle certified'
        ]
        
        certifications = []
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in cert_keywords):
                certifications.append(line.strip())
        
        return certifications[:3]  # Return top 3
    
    def _extract_portfolio_links(self, text: str) -> Dict[str, str]:
        """Extract portfolio links"""
        links = {}
        
        # GitHub
        github_match = re.search(r'github\.com/[a-zA-Z0-9\-_]+', text)
        if github_match:
            links['github'] = f"https://{github_match.group(0)}"
        
        # LinkedIn
        linkedin_match = re.search(r'linkedin\.com/in/[a-zA-Z0-9\-_]+', text)
        if linkedin_match:
            links['linkedin'] = f"https://{linkedin_match.group(0)}"
        
        # Personal website/portfolio
        portfolio_patterns = [
            r'https?://[a-zA-Z0-9\-_]+\.[a-zA-Z]{2,}',
            r'[a-zA-Z0-9\-_]+\.[a-zA-Z]{2,}'
        ]
        
        for pattern in portfolio_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if 'github' not in match and 'linkedin' not in match:
                    if not match.startswith('http'):
                        match = f"https://{match}"
                    links['portfolio'] = match
                    break
        
        return links
    
    def _analyze_indian_context(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Indian context in resume"""
        analysis = {
            'is_indian_education': False,
            'indian_locations': [],
            'indian_terms': [],
            'education_tier': 'Unknown'
        }
        
        # Check education
        for edu in result.get('education', []):
            institution = edu.get('institution', '').lower()
            
            # Check for Indian universities
            for uni in self.indian_universities:
                if uni.lower() in institution:
                    analysis['is_indian_education'] = True
                    
                    # Determine tier
                    if 'iit' in institution or 'iim' in institution:
                        analysis['education_tier'] = 'Premium'
                    elif 'thapar' in institution or 'bits' in institution:
                        analysis['education_tier'] = 'Good'
                    else:
                        analysis['education_tier'] = 'Standard'
                    break
        
        # Check for Indian locations
        locations = ['Punjab', 'Haryana', 'Chandigarh', 'Delhi', 'Mumbai', 
                    'Bangalore', 'Hyderabad', 'Chennai', 'Pune', 'Kolkata']
        
        full_text = result.get('raw_text', '').lower()
        for location in locations:
            if location.lower() in full_text:
                analysis['indian_locations'].append(location)
        
        # Check for Indian terms
        indian_terms = ['lpa', 'ctc', 'fresher', 'b.tech', 'm.tech', 'iit', 'nit']
        for term in indian_terms:
            if term in full_text:
                analysis['indian_terms'].append(term.upper())
        
        return analysis
    
    def _predict_field(self, result: Dict[str, Any]) -> str:
        """Predict career field based on skills and projects"""
        # Get all text for analysis
        skills_text = ' '.join([
            ' '.join(skills) for skills in result.get('skills', {}).values()
        ]).lower()
        
        projects_text = ' '.join([
            p.get('title', '') + ' ' + p.get('description', '') 
            for p in result.get('projects', [])
        ]).lower()
        
        all_text = skills_text + ' ' + projects_text
        
        # Field patterns
        field_patterns = {
            'Web Development': ['html', 'css', 'javascript', 'react', 'vue', 'angular', 
                              'django', 'flask', 'node', 'frontend', 'backend'],
            'Data Science': ['python', 'pandas', 'numpy', 'machine learning', 'deep learning',
                           'tensorflow', 'pytorch', 'data analysis', 'statistics'],
            'Mobile Development': ['android', 'ios', 'flutter', 'react native', 'kotlin', 'swift'],
            'DevOps': ['aws', 'azure', 'docker', 'kubernetes', 'ci/cd', 'terraform', 'ansible'],
            'Software Engineering': ['java', 'c++', 'c#', 'python', 'software', 'algorithm', 'oop']
        }
        
        # Score each field
        field_scores = {}
        for field, keywords in field_patterns.items():
            score = sum(1 for keyword in keywords if keyword in all_text)
            field_scores[field] = score
        
        # Get top field
        if field_scores:
            top_field = max(field_scores.items(), key=lambda x: x[1])
            return top_field[0] if top_field[1] > 0 else "Computer Science"
        
        return "Computer Science"
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result structure"""
        return {
            'name': '',
            'email': '',
            'phone': '',
            'sections': {},
            'education': [],
            'skills': {},
            'projects': [],
            'experience': [],
            'total_experience': 0,
            'experience_level': 'Unknown',
            'certifications': [],
            'portfolio_links': {},
            'indian_context': {},
            'predicted_field': 'Unknown',
            'total_skills': 0,
            'raw_text': ''
        }
    
    def get_summary(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary of parsed resume"""
        return {
            'personal_info': {
                'name': parsed_data.get('name', ''),
                'email': parsed_data.get('email', ''),
                'phone': parsed_data.get('phone', '')
            },
            'education_summary': {
                'count': len(parsed_data.get('education', [])),
                'institutions': [e.get('institution', '') for e in parsed_data.get('education', [])][:3]
            },
            'skills_summary': {
                'total': parsed_data.get('total_skills', 0),
                'categories': list(parsed_data.get('skills', {}).keys())
            },
            'experience_summary': {
                'level': parsed_data.get('experience_level', ''),
                'years': parsed_data.get('total_experience', 0),
                'projects': len(parsed_data.get('projects', []))
            },
            'field': parsed_data.get('predicted_field', 'Unknown')
        }