"""
AI Resume Analyzer with Advanced NLP Features
Revised version with FIXED parser integration and Beautiful UI
"""
import spacy

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()


import streamlit as st

# Page configuration
st.set_page_config(
    page_title="AI Resume Analyzer Pro",
    page_icon='./static/images/logo.jpg',
    layout="wide",
    initial_sidebar_state="expanded"
)

# At the beginning of EnhancedSemanticMatcher.__init__(), add:
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

import pandas as pd
import numpy as np
import base64
import random
import time
import datetime
import json
import sys
import os
import io

# Add custom modules to path
sys.path.append('./nlp_modules')
sys.path.append('./utils')

# Core NLP libraries
from pdfminer.high_level import extract_text as pdf_extract_text
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import TextConverter
from streamlit_tags import st_tags
from PIL import Image
import pymysql
import plotly.express as px
import plotly.graph_objects as go
import nltk

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Import custom modules
from nlp_modules.advanced_analyzer import AdvancedResumeAnalyzer
from nlp_modules.indian_context_processor import IndianContextProcessor
from nlp_modules.semantic_matcher import EnhancedSemanticMatcher
from nlp_modules.resume_parser_enhanced import EnhancedResumeParser  # FIXED PARSER
from utils.preprocessing import TextPreprocessor
from utils.visualization import create_resume_visualizations
from utils.database_handler import DatabaseHandler
from utils.ui_components import *
from Courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos

# Initialize custom modules
@st.cache_resource
def load_nlp_models():
    """Load all NLP models (cached for performance)"""
    analyzer = AdvancedResumeAnalyzer()
    indian_processor = IndianContextProcessor()
    semantic_matcher = EnhancedSemanticMatcher()
    text_preprocessor = TextPreprocessor()
    
    return analyzer, indian_processor, semantic_matcher, text_preprocessor

# Initialize models
analyzer, indian_processor, semantic_matcher, text_preprocessor = load_nlp_models()

def fetch_yt_video(link):
    """Fetch YouTube video title"""
    try:
        import pafy
        video = pafy.new(link)
        return video.title
    except:
        return "Interview Preparation Video"

def get_table_download_link(df, filename, text):
    """Generate download link for dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def pdf_reader(file_path):
    """Extract text from PDF"""
    try:
        return pdf_extract_text(file_path)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_file(file_path):
    """Extract text from different file formats"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.pdf':
            return pdf_extract_text(file_path)
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
        st.error(f"Error extracting text: {e}")
        return ""



def course_recommender(course_list):
    """Recommend courses based on skills"""
    st.subheader("**Courses & Certificates Recommendations 🎓**")
    c = 0
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 5)
    random.shuffle(course_list)
    
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course

def calculate_resume_score_advanced(resume_text, resume_data):
    """Advanced resume scoring with NLP"""
    score_breakdown = {}
    total_score = 0
    
    # 1. Section completeness (30 points)
    sections = ['objective', 'experience', 'education', 'skills', 'projects', 'achievements']
    section_score = 0
    found_sections = []
    
    for section in sections:
        if section in resume_text.lower():
            section_score += 5
            found_sections.append(section)
    
    score_breakdown['section_completeness'] = {
        'score': section_score,
        'max': 30,
        'found_sections': found_sections
    }
    total_score += section_score
    
    # 2. Keyword density and relevance (30 points)
    skills_count = 0
    if 'skills' in resume_data:
        # Count all skills across categories
        for category_skills in resume_data['skills'].values():
            skills_count += len(category_skills)
    
    keyword_score = min(30, skills_count * 2)
    score_breakdown['keyword_relevance'] = {
        'score': keyword_score,
        'max': 30,
        'skills_count': skills_count
    }
    total_score += keyword_score
    
    # 3. Experience quantification (20 points)
    exp_score = 0
    exp_keywords = ['years', 'experience', 'worked', 'developed', 'managed', 'led']
    exp_count = sum(1 for keyword in exp_keywords if keyword in resume_text.lower())
    exp_score = min(20, exp_count * 3)
    
    score_breakdown['experience_quantification'] = {
        'score': exp_score,
        'max': 20,
        'keywords_found': exp_count
    }
    total_score += exp_score
    
    # 4. Action verbs and impact (20 points)
    action_verbs = ['achieved', 'increased', 'reduced', 'developed', 'implemented', 
                    'managed', 'led', 'created', 'improved', 'optimized']
    action_count = sum(1 for verb in action_verbs if verb in resume_text.lower())
    action_score = min(20, action_count * 2)
    
    score_breakdown['action_orientation'] = {
        'score': action_score,
        'max': 20,
        'action_verbs': action_count
    }
    total_score += action_score
    
    return total_score, score_breakdown

def display_skills_analysis(resume_data):
    """Display skills analysis with beautiful badges"""
    st.markdown("### 🛠️ Skills Analysis")
    
    if 'skills' in resume_data and resume_data['skills']:
        skills = resume_data['skills']
        
        category_colors = {
            'Programming': {"color": "#3B82F6", "icon": "💻"},
            'Web Development': {"color": "#8B5CF6", "icon": "🌐"},
            'Data Science': {"color": "#10B981", "icon": "📊"},
            'Tools': {"color": "#F59E0B", "icon": "🛠️"},
            'Frameworks': {"color": "#EC4899", "icon": "⚡"},
            'Other': {"color": "#6B7280", "icon": "📝"}
        }
        
        for category, skill_list in skills.items():
            if skill_list:
                # Get category color
                cat_info = category_colors.get(category, {"color": "#6B7280", "icon": "📝"})
                
                # Category header
                st.markdown(f"""
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin: 1.5rem 0 1rem;
                    padding-bottom: 0.5rem;
                    border-bottom: 2px solid {cat_info['color']}30;
                ">
                    <span style="font-size: 1.5rem;">{cat_info['icon']}</span>
                    <h4 style="margin: 0; color: {cat_info['color']}; font-weight: 600;">
                        {category} ({len(skill_list)})
                    </h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Display skills as badges
                badge_html = create_skill_badges(skill_list, {
                    'python': '#3B82F6',
                    'javascript': '#F59E0B',
                    'react': '#61DAFB',
                    'html': '#E34F26',
                    'css': '#1572B6',
                    'aws': '#FF9900',
                    'docker': '#2496ED',
                    'git': '#F05032',
                    'machine learning': '#10B981',
                    'sql': '#336791',
                    'node': '#339933',
                    'java': '#007396',
                    'c++': '#00599C',
                    'docker': '#2496ED',
                    'kubernetes': '#326CE5',
                    'angular': '#DD0031',
                    'vue': '#4FC08D',
                    'typescript': '#3178C6',
                    'mongodb': '#47A248',
                    'postgresql': '#336791',
                    'redis': '#DC382D',
                    'graphql': '#E10098',
                    'spring': '#6DB33F',
                    'django': '#092E20',
                    'flask': '#000000',
                    'fastapi': '#009688',
                    'tensorflow': '#FF6F00',
                    'pytorch': '#EE4C2C',
                    'pandas': '#150458',
                    'numpy': '#013243',
                    'matplotlib': '#11557C',
                    'seaborn': '#3776AB',
                    'scikit-learn': '#F7931E',
                    'spark': '#E25A1C',
                    'hadoop': '#66CCFF',
                    'airflow': '#017CEE',
                    'kafka': '#231F20',
                    'elasticsearch': '#005571',
                    'jenkins': '#D24939',
                    'terraform': '#623CE4',
                    'ansible': '#EE0000',
                    'prometheus': '#E6522C',
                    'grafana': '#F46800',
                    'splunk': '#000000',
                    'tableau': '#E97627',
                    'power bi': '#F2C811',
                    'excel': '#217346',
                    'word': '#2B579A',
                    'powerpoint': '#D24726',
                    'outlook': '#0072C6',
                    'jira': '#0052CC',
                    'confluence': '#172B4D',
                    'slack': '#4A154B',
                    'teams': '#6264A7',
                    'zoom': '#2D8CFF',
                    'figma': '#F24E1E',
                    'adobe xd': '#FF61F6',
                    'sketch': '#F7B500',
                    'illustrator': '#FF9A00',
                    'photoshop': '#31A8FF',
                    'premiere pro': '#EA77FF',
                    'after effects': '#9999FF',
                    'blender': '#F5792A',
                    'unity': '#000000',
                    'unreal': '#0E1128'
                })
                
                st.markdown(badge_html, unsafe_allow_html=True)
    else:
        st.info("No skills extracted. Try adding a 'SKILLS' section to your resume.")

def display_career_path(resume_data):
    """Display career path recommendations"""
    st.write("### 🎯 Career Path Analysis")
    
    experience_level = resume_data.get('experience_level', 'Unknown')
    total_experience = resume_data.get('total_experience', 0)
    predicted_field = resume_data.get('predicted_field', 'Computer Science')
    
    # Create beautiful career path cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(create_metric_card(
            "Current Level", 
            experience_level, 
            icon="📊", 
            color="#3B82F6"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_card(
            "Experience", 
            f"{total_experience} years", 
            icon="⏳", 
            color="#10B981"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_metric_card(
            "Predicted Field", 
            predicted_field, 
            icon="🎯", 
            color="#8B5CF6"
        ), unsafe_allow_html=True)
    
    # Career progression based on experience
    st.write("### 🚀 Growth Path")
    
    if total_experience == 0:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #3B82F615 0%, #10B98115 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            border-left: 5px solid #3B82F6;
        ">
            <h4 style="color: #3B82F6; margin-top: 0;">Entry Level (0-2 years)</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
                <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 3px 10px rgba(0,0,0,0.05);">
                    <h5 style="color: #10B981; margin-top: 0;">🏗️ Build Portfolio</h5>
                    <ul style="margin: 0.5rem 0; padding-left: 1.2rem;">
                        <li>Create 2-3 real-world projects</li>
                        <li>Contribute to open-source</li>
                        <li>Build a personal website</li>
                    </ul>
                </div>
                <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 3px 10px rgba(0,0,0,0.05);">
                    <h5 style="color: #F59E0B; margin-top: 0;">🎯 Get Experience</h5>
                    <ul style="margin: 0.5rem 0; padding-left: 1.2rem;">
                        <li>Apply for internships</li>
                        <li>Freelance projects</li>
                        <li>Hackathons & competitions</li>
                    </ul>
                </div>
                <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 3px 10px rgba(0,0,0,0.05);">
                    <h5 style="color: #8B5CF6; margin-top: 0;">📚 Learn Skills</h5>
                    <ul style="margin: 0.5rem 0; padding-left: 1.2rem;">
                        <li>Master core technologies</li>
                        <li>Get industry certifications</li>
                        <li>Build communication skills</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif total_experience < 2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #10B98115 0%, #F59E0B15 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            border-left: 5px solid #10B981;
        ">
            <h4 style="color: #10B981; margin-top: 0;">Junior Level (1-3 years)</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
                <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 3px 10px rgba(0,0,0,0.05);">
                    <h5 style="color: #3B82F6; margin-top: 0;">⚡ Take Ownership</h5>
                    <ul style="margin: 0.5rem 0; padding-left: 1.2rem;">
                        <li>Lead small features/projects</li>
                        <li>Improve existing systems</li>
                        <li>Document processes</li>
                    </ul>
                </div>
                <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 3px 10px rgba(0,0,0,0.05);">
                    <h5 style="color: #EC4899; margin-top: 0;">👨‍💼 Learn Leadership</h5>
                    <ul style="margin: 0.5rem 0; padding-left: 1.2rem;">
                        <li>Mentor interns/new hires</li>
                        <li>Improve team processes</li>
                        <li>Participate in planning</li>
                    </ul>
                </div>
                <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 3px 10px rgba(0,0,0,0.05);">
                    <h5 style="color: #8B5CF6; margin-top: 0;">📈 Specialize</h5>
                    <ul style="margin: 0.5rem 0; padding-left: 1.2rem;">
                        <li>Choose a tech stack focus</li>
                        <li>Get advanced certifications</li>
                        <li>Build domain expertise</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #F59E0B15 0%, #EF444415 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            border-left: 5px solid #F59E0B;
        ">
            <h4 style="color: #F59E0B; margin-top: 0;">Senior Level (3+ years)</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
                <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 3px 10px rgba(0,0,0,0.05);">
                    <h5 style="color: #3B82F6; margin-top: 0;">👥 Mentor Others</h5>
                    <ul style="margin: 0.5rem 0; padding-left: 1.2rem;">
                        <li>Guide junior developers</li>
                        <li>Create learning materials</li>
                        <li>Build team culture</li>
                    </ul>
                </div>
                <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 3px 10px rgba(0,0,0,0.05);">
                    <h5 style="color: #10B981; margin-top: 0;">🏗️ Architecture</h5>
                    <ul style="margin: 0.5rem 0; padding-left: 1.2rem;">
                        <li>Design complex systems</li>
                        <li>Make tech stack decisions</li>
                        <li>Ensure scalability</li>
                    </ul>
                </div>
                <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 3px 10px rgba(0,0,0,0.05);">
                    <h5 style="color: #8B5CF6; margin-top: 0;">🎯 Niche Expertise</h5>
                    <ul style="margin: 0.5rem 0; padding-left: 1.2rem;">
                        <li>Become a domain expert</li>
                        <li>Speak at conferences</li>
                        <li>Write technical content</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_ats_score(resume_text, resume_data):
    """Display ATS score breakdown with beautiful UI"""
    score, breakdown = calculate_resume_score_advanced(resume_text, resume_data)
    
    # Determine color based on score
    if score >= 80:
        color = "#10B981"
        icon = "🎉"
        label = "Excellent"
    elif score >= 60:
        color = "#F59E0B"
        icon = "👍"
        label = "Good"
    else:
        color = "#EF4444"
        icon = "📈"
        label = "Needs Work"
    
    # Main ATS Score Card
    st.markdown(f"""
    <div style="
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
        border: 2px solid {color}30;
    ">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="font-size: 3.5rem; font-weight: 800; color: {color}; margin-bottom: 0.5rem;">
            {score}/100
        </div>
        <div style="font-size: 1.3rem; color: #666; margin-bottom: 1.5rem; font-weight: 600;">
            ATS Score: {label}
        </div>
        <div style="
            width: 80%;
            margin: 0 auto;
            height: 15px;
            background: #E5E7EB;
            border-radius: 10px;
            overflow: hidden;
        ">
            <div style="
                width: {score}%;
                height: 100%;
                background: linear-gradient(90deg, {color}, {color}80);
                border-radius: 10px;
                transition: width 1s ease-in-out;
            "></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Score breakdown in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        skills_count = breakdown['keyword_relevance']['skills_count']
        st.markdown(create_metric_card(
            "Skills Found", 
            f"{skills_count}", 
            icon="🛠️", 
            color="#3B82F6",
            change=10 if skills_count > 10 else -10
        ), unsafe_allow_html=True)
    
    with col2:
        sections_found = len(breakdown['section_completeness']['found_sections'])
        st.markdown(create_metric_card(
            "Sections", 
            f"{sections_found}/6", 
            icon="📋", 
            color="#10B981",
            change=10 if sections_found >= 4 else -10
        ), unsafe_allow_html=True)
    
    with col3:
        action_verbs = breakdown['action_orientation']['action_verbs']
        st.markdown(create_metric_card(
            "Action Verbs", 
            f"{action_verbs}", 
            icon="⚡", 
            color="#8B5CF6",
            change=10 if action_verbs > 5 else -10
        ), unsafe_allow_html=True)
    
    # Detailed breakdown in expander
    with st.expander("📊 View Detailed Score Breakdown", expanded=False):
        categories = {
            'section_completeness': ('Section Completeness', '📋', '#10B981'),
            'keyword_relevance': ('Keyword Relevance', '🎯', '#3B82F6'),
            'experience_quantification': ('Experience Quantification', '⏳', '#F59E0B'),
            'action_orientation': ('Action Orientation', '⚡', '#8B5CF6')
        }
        
        for category_key, (category_name, icon, color) in categories.items():
            details = breakdown[category_key]
            score_val = details['score']
            max_val = details['max']
            
            # Progress bar with label
            st.markdown(f"**{icon} {category_name}:** {score_val}/{max_val}")
            st.markdown(create_progress_bar(
                f"{category_name} ({score_val}/{max_val})",
                int((score_val/max_val)*100),
                color
            ), unsafe_allow_html=True)
            
            # Show details
            for key, value in details.items():
                if key not in ['score', 'max']:
                    if isinstance(value, list):
                        st.write(f"  **{key.replace('_', ' ').title()}:** {', '.join(value)}")
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            st.write(f"  **{sub_key.replace('_', ' ').title()}:** {sub_value}")
                    else:
                        st.write(f"  **{key.replace('_', ' ').title()}:** {value}")
            st.write("---")

def display_enhanced_indian_context(resume_text, resume_data):
    """Display SUPER enhanced Indian context analysis with beautiful UI"""
    st.markdown("### 🇮🇳 DEEP INDIAN MARKET INSIGHTS")
    
    # Initialize enhanced processor
    indian_processor = IndianContextProcessor()
    
    # Get comprehensive analysis
    analysis = indian_processor.analyze_indian_context(resume_text)
    
    # Display in tabs for better organization
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Market Position", 
        "💰 Salary Insights", 
        "🛠️ Skills Analysis", 
        "📈 Growth Potential",
        "📊 Full Report"
    ])
    
    with tab1:
        # Market Position Analysis
        market_position = analysis.get('market_position', {})
        score = market_position.get('score', 0)
        position = market_position.get('position', 'Unknown')
        
        # Determine color
        if score >= 30:
            color = "#10B981"
            icon = "🏆"
        elif score >= 20:
            color = "#F59E0B"
            icon = "📈"
        else:
            color = "#EF4444"
            icon = "📊"
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(create_metric_card(
                "Market Score", 
                f"{score}/40", 
                icon=icon, 
                color=color
            ), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card(
                "Market Position", 
                position, 
                icon="🎯", 
                color="#3B82F6"
            ), unsafe_allow_html=True)
        
        # Score breakdown
        components = market_position.get('components', {})
        st.write("**Score Breakdown:**")
        for component, comp_score in components.items():
            component_name = component.replace('_', ' ').title()
            st.markdown(create_progress_bar(
                f"{component_name} ({comp_score}/10)",
                comp_score*10,
                "#8B5CF6"
            ), unsafe_allow_html=True)
    
    with tab2:
        # Salary Insights
        salary = analysis.get('salary_analysis', {})
        market_comparison = salary.get('market_comparison', {})
        
        st.write("**💰 Salary Benchmarks:**")
        cols = st.columns(2)
        with cols[0]:
            service_salary = market_comparison.get('service_company_benchmark', 'N/A')
            st.markdown(create_metric_card(
                "Service Companies", 
                service_salary, 
                icon="🏢", 
                color="#3B82F6"
            ), unsafe_allow_html=True)
        with cols[1]:
            product_salary = market_comparison.get('product_company_benchmark', 'N/A')
            st.markdown(create_metric_card(
                "Product Companies", 
                product_salary, 
                icon="💻", 
                color="#10B981"
            ), unsafe_allow_html=True)
        
        # Negotiation leverage
        leverage = salary.get('negotiation_leverage', [])
        if leverage:
            st.write("**💪 Negotiation Power:**")
            for point in leverage[:3]:
                st.info(f"• {point}")
    
    with tab3:
        # Skills Analysis
        skills = analysis.get('skills_context', {})
        demand = skills.get('demand_analysis', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            high_demand = demand.get('high_demand_skills_found', 0)
            st.markdown(create_metric_card(
                "High Demand", 
                str(high_demand), 
                icon="🔥", 
                color="#EF4444"
            ), unsafe_allow_html=True)
        with col2:
            emerging = demand.get('emerging_skills_found', 0)
            st.markdown(create_metric_card(
                "Emerging", 
                str(emerging), 
                icon="🌱", 
                color="#10B981"
            ), unsafe_allow_html=True)
        with col3:
            stable = demand.get('stable_skills_found', 0)
            st.markdown(create_metric_card(
                "Stable", 
                str(stable), 
                icon="📊", 
                color="#3B82F6"
            ), unsafe_allow_html=True)
        
        # Premium skills
        salary_impact = skills.get('salary_impact', {})
        if salary_impact.get('premium_skills'):
            st.write("**💎 Premium Skills Found:**")
            for skill_info in salary_impact['premium_skills'][:3]:
                st.success(f"**{skill_info['skill'].title()}**: {skill_info['premium']}")
    
    with tab4:
        # Growth Potential
        growth = analysis.get('career_growth', {})
        
        # Timeline visualization
        st.write("**🚀 Career Trajectory Timeline**")
        
        # Short term
        if growth.get('short_term_opportunities'):
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #3B82F615 0%, #10B98115 100%);
                border-radius: 15px;
                padding: 1.5rem;
                margin: 1rem 0;
                border-left: 5px solid #3B82F6;
            ">
                <h4 style="color: #3B82F6; margin-top: 0;">📅 Short-term (0-2 years)</h4>
            """, unsafe_allow_html=True)
            for opp in growth['short_term_opportunities'][:2]:
                st.write(f"• {opp}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Mid term
        if growth.get('mid_term_trajectory'):
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #F59E0B15 0%, #EC489915 100%);
                border-radius: 15px;
                padding: 1.5rem;
                margin: 1rem 0;
                border-left: 5px solid #F59E0B;
            ">
                <h4 style="color: #F59E0B; margin-top: 0;">📅 Mid-term (2-5 years)</h4>
            """, unsafe_allow_html=True)
            for traj in growth['mid_term_trajectory'][:2]:
                st.write(f"• {traj}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Growth boosters
        if growth.get('growth_boosters'):
            st.write("**📈 Acceleration Opportunities:**")
            for booster in growth['growth_boosters'][:3]:
                st.success(f"• {booster}")
    
    with tab5:
        # Full Report
        report = indian_processor.generate_insights_report(analysis)
        st.code(report, language='markdown')
        
        # Download button for report
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Download Insights Report", type="primary", use_container_width=True):
                st.download_button(
                    label="Click to Download",
                    data=report,
                    file_name=f"indian_market_insights_{datetime.datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
        with col2:
            if st.button("📧 Share Report", use_container_width=True):
                st.info("Share feature coming soon!")
    
    # Competitive Analysis (always visible)
    st.markdown("---")
    st.write("### ⚔️ Competitive Analysis")
    
    competitive = analysis.get('competitive_analysis', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #10B98115 0%, #3B82F615 100%);
            border-radius: 15px;
            padding: 1.5rem;
            height: 100%;
            border-left: 5px solid #10B981;
        ">
            <h4 style="color: #10B981; margin-top: 0;">✅ Your Strengths</h4>
        """, unsafe_allow_html=True)
        strengths = competitive.get('strengths', [])
        for strength in strengths[:4]:
            st.write(f"• {strength}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #F59E0B15 0%, #EF444415 100%);
            border-radius: 15px;
            padding: 1.5rem;
            height: 100%;
            border-left: 5px solid #F59E0B;
        ">
            <h4 style="color: #F59E0B; margin-top: 0;">📝 Market Gaps</h4>
        """, unsafe_allow_html=True)
        gaps = competitive.get('gaps', [])
        for gap in gaps[:4]:
            st.write(f"• {gap}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Unique Differentiators
    if competitive.get('unique_selling_points'):
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #8B5CF615 0%, #EC489915 100%);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            border-left: 5px solid #8B5CF6;
        ">
            <h4 style="color: #8B5CF6; margin-top: 0;">💎 Unique Selling Points</h4>
        """, unsafe_allow_html=True)
        for usp in competitive['unique_selling_points'][:3]:
            st.info(f"• {usp}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Strategic Recommendations
    st.markdown("---")
    st.write("### 🚀 Strategic Action Plan")
    
    recommendations = analysis.get('recommendations', [])
    if recommendations:
        for i, rec in enumerate(recommendations[:5], 1):
            st.markdown(f"""
            <div style="
                background: white;
                border-radius: 12px;
                padding: 1rem 1.5rem;
                margin: 0.5rem 0;
                border-left: 4px solid #6366f1;
                box-shadow: 0 3px 10px rgba(0,0,0,0.05);
            ">
                <div style="display: flex; align-items: center;">
                    <div style="
                        background: #6366f1;
                        color: white;
                        width: 30px;
                        height: 30px;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin-right: 12px;
                        font-weight: bold;
                    ">
                        {i}
                    </div>
                    <span style="font-weight: 600; color: #374151;">{rec}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Location-based opportunities
    location = analysis.get('location_analysis', {})
    if location.get('opportunity_hotspots'):
        st.write("**📍 Top Location Opportunities:**")
        cols = st.columns(2)
        for i, hotspot in enumerate(location['opportunity_hotspots'][:2]):
            hub_info = hotspot.get('info', {})
            with cols[i]:
                st.markdown(create_metric_card(
                    hotspot['hub'],
                    f"+{hub_info.get('avg_salary_premium', 0)}% salary",
                    icon="📍",
                    color="#EC4899"
                ), unsafe_allow_html=True)

def display_recommendations(resume_data):
    """Display personalized recommendations with beautiful UI"""
    st.markdown("### 💡 Personalized Recommendations")
    
    predicted_field = resume_data.get('predicted_field', '').lower()
    
    course_map = {
        'data science': ds_course,
        'web development': web_course,
        'android development': android_course,
        'ios development': ios_course,
        'ui/ux': uiux_course
    }
    
    # Find best matching field
    selected_courses = None
    for field, courses in course_map.items():
        if field in predicted_field:
            selected_courses = courses
            break
    
    if selected_courses is None:
        selected_courses = web_course
    
    # Display courses in beautiful cards
    st.write("#### 🎓 Recommended Courses")
    rec_course = course_recommender(selected_courses)
    
    # Interview tips with beautiful card
    st.markdown("---")
    st.write("### 🎤 Interview Preparation")
    
    if interview_videos:
        interview_vid = random.choice(interview_videos)
        try:
            int_vid_title = fetch_yt_video(interview_vid)
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
                border-radius: 15px;
                padding: 1.5rem;
                margin: 1rem 0;
                color: white;
            ">
                <h4 style="color: white; margin-top: 0;">🎬 Recommended Interview Prep</h4>
                <p style="margin-bottom: 1rem;"><strong>{int_vid_title}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            st.video(interview_vid)
        except:
            st.info("Interview preparation resources")

def store_analysis_results(db_handler, resume_data):
    """Store analysis results in database"""
    try:
        analysis_data = {
            'name': resume_data.get('name', ''),
            'email': resume_data.get('email', ''),
            'skills': json.dumps(resume_data.get('skills', {})),
            'predicted_field': resume_data.get('predicted_field', 'Unknown'),
            'total_experience': resume_data.get('total_experience', 0),
            'experience_level': resume_data.get('experience_level', 'Unknown'),
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        db_handler.insert_analysis(analysis_data)
        st.success("✅ Analysis saved to database!")
    except Exception as e:
        st.error(f"Could not save to database: {str(e)}")

def display_education_info(resume_data):
    """Display education information with beautiful cards"""
    st.write("### 🎓 Education")
    
    education = resume_data.get('education', [])
    
    if education:
        cols = st.columns(min(3, len(education)))
        for i, edu in enumerate(education):
            with cols[i % len(cols)]:
                st.markdown(f"""
                <div style="
                    background: white;
                    border-radius: 15px;
                    padding: 1.5rem;
                    margin: 0.5rem 0;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
                    border-top: 4px solid #3B82F6;
                    height: 100%;
                ">
                    <h4 style="color: #3B82F6; margin-top: 0; font-size: 1.1rem;">
                        {edu.get('institution', 'Unknown Institution')}
                    </h4>
                """, unsafe_allow_html=True)
                
                if edu.get('degree'):
                    st.write(f"**Degree:** {edu['degree']}")
                if edu.get('year'):
                    st.write(f"**Year:** {edu['year']}")
                if edu.get('gpa'):
                    st.write(f"**GPA:** {edu['gpa']}")
                if edu.get('details'):
                    st.write(f"**Details:** {edu['details'][:100]}...")
                
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No education information found. Add an 'EDUCATION' section to your resume.")

def resume_jd_matching_mode():
    """Resume vs Job Description matching mode with beautiful UI"""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
    ">
        <h2 style="color: white; margin: 0;">📋 Resume vs Job Description Matching</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0;">
            Upload your resume and job description for detailed compatibility analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📄 Upload Resume")
        st.markdown("""
        <div style="
            background: white;
            border-radius: 15px;
            padding: 2rem;
            margin: 1rem 0;
            text-align: center;
            border: 3px dashed rgba(99, 102, 241, 0.3);
        ">
            <span style="font-size: 3rem;">📄</span>
            <p style="color: #666; margin: 1rem 0;">Drop your resume here or click to upload</p>
        </div>
        """, unsafe_allow_html=True)
        resume_file = st.file_uploader("Choose Resume", type=["pdf", "docx", "txt"], key="resume_jd", label_visibility="collapsed")
    
    with col2:
        st.markdown("### 📋 Upload Job Description")
        st.markdown("""
        <div style="
            background: white;
            border-radius: 15px;
            padding: 2rem;
            margin: 1rem 0;
            text-align: center;
            border: 3px dashed rgba(139, 92, 246, 0.3);
        ">
            <span style="font-size: 3rem;">📋</span>
            <p style="color: #666; margin: 1rem 0;">Drop JD here or click to upload</p>
        </div>
        """, unsafe_allow_html=True)
        jd_file = st.file_uploader("Choose Job Description", type=["pdf", "txt"], key="jd_file", label_visibility="collapsed")
        
        st.markdown("### **OR**")
        jd_text = st.text_area("Paste Job Description text:", height=200, 
                             placeholder="Copy and paste the job description here...",
                             help="Paste the complete job description for analysis")
    
    if resume_file and (jd_file or jd_text):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Match Resume with JD", type="primary", use_container_width=True):
                with st.spinner("🔍 Analyzing match with enhanced skill extraction..."):
                    resume_path = f'./Uploaded_Resumes/jd_match_{resume_file.name}'
                    os.makedirs('./Uploaded_Resumes', exist_ok=True)
                    
                    with open(resume_path, "wb") as f:
                        f.write(resume_file.getbuffer())
                    
                    resume_text = extract_text_from_file(resume_path)
                    
                    if jd_file:
                        jd_path = f'./Uploaded_Resumes/{jd_file.name}'
                        with open(jd_path, "wb") as f:
                            f.write(jd_file.getbuffer())
                        jd_text_content = extract_text_from_file(jd_path)
                    else:
                        jd_text_content = jd_text
                    
                    if not jd_text_content.strip():
                        st.error("Please provide a job description")
                        return
                    
                    try:
                        match_results = semantic_matcher.match_resume_jd(resume_text, jd_text_content)
                        display_enhanced_match_results(match_results)
                    except Exception as e:
                        st.error(f"Error during matching: {e}")
                        display_simple_match_results(resume_text, jd_text_content)

def display_simple_match_results(resume_text, jd_text):
    """Simple fallback matching display"""
    resume_words = set(resume_text.lower().split())
    jd_words = set(jd_text.lower().split())
    
    matched_words = resume_words.intersection(jd_words)
    total_jd_words = len(jd_words)
    
    if total_jd_words > 0:
        match_percentage = (len(matched_words) / total_jd_words) * 100
    else:
        match_percentage = 0
    
    # Display with beautiful UI
    display_enhanced_metrics({'match_score': match_percentage})
    
    if matched_words:
        important_words = [word for word in matched_words if len(word) > 3 and word not in ['with', 'have', 'from', 'this', 'that', 'will']]
        
        st.markdown("### ✅ Matched Keywords")
        badge_html = ""
        for word in important_words[:15]:
            badge_html += f"""
            <span style="
                display: inline-block;
                background: linear-gradient(135deg, #10B981 0%, #059669 100%);
                color: white;
                padding: 6px 14px;
                border-radius: 20px;
                margin: 4px;
                font-size: 0.9rem;
                font-weight: 500;
            ">
                {word.title()}
            </span>
            """
        
        st.markdown(badge_html, unsafe_allow_html=True)

def display_enhanced_match_results(match_results):
    """Display enhanced matching results with beautiful UI"""
    # Display main metrics
    display_enhanced_metrics(match_results)
    
    # Experience analysis
    exp_analysis = match_results.get('experience_analysis', {})
    if exp_analysis:
        st.markdown("### 📈 Experience Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            required = exp_analysis.get('jd_required', 0)
            st.markdown(create_metric_card(
                "Job Requires", 
                f"{required} years", 
                icon="🎯", 
                color="#3B82F6"
            ), unsafe_allow_html=True)
        
        with col2:
            has_exp = exp_analysis.get('resume_has', 0)
            st.markdown(create_metric_card(
                "You Have", 
                f"{has_exp} years", 
                icon="📊", 
                color="#10B981" if has_exp >= required else "#F59E0B"
            ), unsafe_allow_html=True)
        
        with col3:
            gap = exp_analysis.get('gap_years', 0)
            color = "#10B981" if gap <= 0 else "#EF4444"
            icon = "✅" if gap <= 0 else "⚠️"
            st.markdown(create_metric_card(
                "Experience Gap", 
                f"{abs(gap)} years {'short' if gap > 0 else 'met'}", 
                icon=icon, 
                color=color
            ), unsafe_allow_html=True)
    
    # Skills analysis in tabs
    st.markdown("### 🛠️ Skills Analysis")
    
    tab1, tab2 = st.tabs(["✅ Your Strengths", "📝 Skills to Improve"])
    
    with tab1:
        strengths = match_results.get('strengths', [])
        if strengths:
            st.markdown(f"**Matched Skills ({len(strengths)}):**")
            
            # Categorize skills
            skill_categories = {}
            for skill in strengths:
                skill_lower = skill.lower()
                if any(tech in skill_lower for tech in ['react', 'angular', 'vue', 'html', 'css', 'javascript', 'typescript']):
                    category = "Frontend"
                elif any(tech in skill_lower for tech in ['node', 'python', 'java', 'spring', 'django', 'express', 'flask', 'fastapi']):
                    category = "Backend"
                elif any(tech in skill_lower for tech in ['aws', 'docker', 'kubernetes', 'devops', 'azure', 'gcp', 'terraform']):
                    category = "DevOps/Cloud"
                elif any(tech in skill_lower for tech in ['sql', 'mongodb', 'postgresql', 'mysql', 'database', 'redis']):
                    category = "Database"
                elif any(tech in skill_lower for tech in ['git', 'agile', 'scrum', 'jira', 'confluence']):
                    category = "Tools/Methodologies"
                elif any(tech in skill_lower for tech in ['machine learning', 'ai', 'data science', 'tensorflow', 'pytorch']):
                    category = "Data Science/AI"
                else:
                    category = "Other Skills"
                
                if category not in skill_categories:
                    skill_categories[category] = []
                skill_categories[category].append(skill.title())
            
            # Display by category
            for category, skills in skill_categories.items():
                with st.expander(f"{category} ({len(set(skills))} skills)", expanded=True):
                    cols = st.columns(2)
                    for i, skill in enumerate(sorted(set(skills))):
                        col_idx = i % 2
                        with cols[col_idx]:
                            st.markdown(f"""
                            <div style="
                                background: #f0f9ff;
                                border-radius: 10px;
                                padding: 8px 12px;
                                margin: 4px 0;
                                border-left: 4px solid #3B82F6;
                            ">
                                {skill}
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.info("No matched skills found")
    
    with tab2:
        missing_skills = match_results.get('missing_skills', [])
        if missing_skills:
            st.markdown(f"**Missing Skills ({len(missing_skills)}):**")
            
            # Group missing skills by priority
            high_priority = []
            medium_priority = []
            low_priority = []
            
            for skill in missing_skills:
                skill_lower = skill.lower()
                # Define priority based on skill type
                if any(tech in skill_lower for tech in ['react', 'python', 'aws', 'docker', 'sql']):
                    high_priority.append(skill)
                elif any(tech in skill_lower for tech in ['javascript', 'node', 'mongodb', 'git']):
                    medium_priority.append(skill)
                else:
                    low_priority.append(skill)
            
            # Display by priority
            if high_priority:
                st.markdown("#### 🔴 High Priority")
                for skill in sorted(set(high_priority)):
                    st.error(f"**{skill.title()}** - Critical for this role")
            
            if medium_priority:
                st.markdown("#### 🟡 Medium Priority")
                for skill in sorted(set(medium_priority)):
                    st.warning(f"**{skill.title()}** - Important but can be learned")
            
            if low_priority:
                st.markdown("#### 🟢 Low Priority")
                for skill in sorted(set(low_priority)):
                    st.info(f"**{skill.title()}** - Good to have")
        else:
            st.success("✅ All required skills are present!")
    
    # Recommendations
    recommendations = match_results.get('recommendations', [])
    if recommendations:
        st.markdown("### 💡 Personalized Recommendations")
        
        for i, rec in enumerate(recommendations[:3], 1):
            rec_type = rec.get('type', 'General').title()
            message = rec.get('message', '')
            action = rec.get('action', '')
            
            st.markdown(create_recommendation_card(
                rec_type,
                message,
                action,
                i
            ), unsafe_allow_html=True)

def skill_gap_analysis_mode():
    """Skill gap analysis mode with beautiful UI"""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
    ">
        <h2 style="color: white; margin: 0;">🔧 Skill Gap Analysis</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0;">
            Identify skill gaps and create a personalized learning plan
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_role = st.selectbox(
            "🎯 Select Target Role:",
            ["Data Scientist", "ML Engineer", "Web Developer", 
             "DevOps Engineer", "Data Analyst", "Product Manager", "Software Engineer"],
            help="Choose the role you want to prepare for"
        )
    
    with col2:
        st.markdown("### 📄 Upload Current Resume")
        st.markdown("""
        <div style="
            background: white;
            border-radius: 15px;
            padding: 2rem;
            margin: 1rem 0;
            text-align: center;
            border: 3px dashed rgba(16, 185, 129, 0.3);
        ">
            <span style="font-size: 3rem;">📄</span>
            <p style="color: #666; margin: 1rem 0;">Upload your current resume for analysis</p>
        </div>
        """, unsafe_allow_html=True)
        current_resume = st.file_uploader("Upload Current Resume", type=["pdf", "docx", "txt"], label_visibility="collapsed")
    
    if current_resume and target_role:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Analyze Skill Gaps", type="primary", use_container_width=True):
                with st.spinner("🔍 Analyzing skill gaps..."):
                    resume_path = f'./Uploaded_Resumes/gap_analysis_{current_resume.name}'
                    os.makedirs('./Uploaded_Resumes', exist_ok=True)
                    
                    with open(resume_path, "wb") as f:
                        f.write(current_resume.getbuffer())
                    
                    parser = EnhancedResumeParser()
                    resume_data = parser.parse_resume(resume_path)
                    
                    current_skills_set = set()
                    if 'skills' in resume_data:
                        for category_skills in resume_data['skills'].values():
                            current_skills_set.update([str(s).lower().strip() for s in category_skills])
                    
                    required_skills = get_required_skills(target_role)
                    matched_skills = required_skills.intersection(current_skills_set)
                    skill_gaps = required_skills - current_skills_set
                    
                    display_skill_gaps(target_role, matched_skills, skill_gaps)

def get_required_skills(role):
    """Get required skills for a role (all lowercase)"""
    role_skills = {
        "Data Scientist": {
            "python", "machine learning", "statistics", "sql", 
            "data visualization", "data analysis", "pandas", "numpy",
            "matplotlib", "seaborn", "scikit-learn", "tensorflow", "pytorch"
        },
        "ML Engineer": {
            "python", "machine learning", "deep learning", "tensorflow", 
            "pytorch", "docker", "aws", "mlops", "data pipelines",
            "kubernetes", "ci/cd", "airflow", "fastapi"
        },
        "Web Developer": {
            "javascript", "html", "css", "react", "node.js", 
            "python", "django", "flask", "rest api", "git",
            "typescript", "vue.js", "angular", "mongodb", "express.js"
        },
        "DevOps Engineer": {
            "docker", "kubernetes", "aws", "ci/cd", "linux", 
            "python", "terraform", "jenkins", "git", "monitoring",
            "ansible", "prometheus", "grafana", "bash scripting"
        },
        "Data Analyst": {
            "sql", "excel", "python", "tableau", "power bi", 
            "statistics", "data visualization", "pandas",
            "numpy", "matplotlib", "seaborn", "business intelligence"
        },
        "Product Manager": {
            "product strategy", "user research", "agile", 
            "data analysis", "stakeholder management", "roadmapping",
            "market research", "jira", "confluence", "user stories"
        },
        "Software Engineer": {
            "python", "java", "javascript", "c++", "sql", 
            "git", "algorithms", "data structures", "oop",
            "system design", "docker", "aws", "testing", "debugging"
        }
    }
    
    return role_skills.get(role, set())

def display_skill_gaps(target_role, matched_skills, skill_gaps):
    """Display skill gap analysis results with beautiful UI"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
    ">
        <h2 style="color: white; margin: 0;">🔍 Skill Gap Analysis for {target_role}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Convert to lowercase for consistency
    matched_skills_lower = {s.lower() for s in matched_skills}
    skill_gaps_lower = {s.lower() for s in skill_gaps}
    total_skills = len(matched_skills_lower) + len(skill_gaps_lower)
    
    # Metrics in beautiful cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(create_metric_card(
            "Total Required Skills", 
            str(total_skills), 
            icon="📊", 
            color="#3B82F6"
        ), unsafe_allow_html=True)
    
    with col2:
        match_percentage = (len(matched_skills_lower) / total_skills * 100) if total_skills > 0 else 0
        st.markdown(create_metric_card(
            "Your Coverage", 
            f"{match_percentage:.1f}%", 
            icon="✅", 
            color="#10B981",
            change=round(match_percentage - 50)
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_metric_card(
            "Skills to Learn", 
            str(len(skill_gaps_lower)), 
            icon="📚", 
            color="#F59E0B"
        ), unsafe_allow_html=True)
    
    # Skills visualization in columns
    st.markdown("### 📊 Skills Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # FIX: Use f-string to insert actual value
        matched_count = len(matched_skills_lower)
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #10B98115 0%, #3B82F615 100%);
            border-radius: 15px;
            padding: 1.5rem;
            height: 100%;
            border-left: 5px solid #10B981;
        ">
            <h4 style="color: #10B981; margin-top: 0;">✅ Skills You Have ({matched_count})</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Display matched skills as badges
        badge_html = ""
        for skill in sorted(matched_skills_lower):
            badge_html += create_skill_badge(skill.title(), "programming")
        
        st.markdown(badge_html, unsafe_allow_html=True)
    
    with col2:
        # FIX: Use f-string to insert actual value
        gaps_count = len(skill_gaps_lower)
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #F59E0B15 0%, #EF444415 100%);
            border-radius: 15px;
            padding: 1.5rem;
            height: 100%;
            border-left: 5px solid #F59E0B;
        ">
            <h4 style="color: #F59E0B; margin-top: 0;">📚 Skills to Learn ({gaps_count})</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Display missing skills as badges
        badge_html = ""
        for skill in sorted(skill_gaps_lower):
            badge_html += create_skill_badge(skill.title(), "data")
        
        st.markdown(badge_html, unsafe_allow_html=True)
    
    # ... rest of the function remains the same
    
    
    # Learning plan
    st.markdown("### 🎓 Personalized Learning Plan")
    
    if skill_gaps_lower:
        # Create tabs for top 3 missing skills
        top_skills = sorted(skill_gaps_lower)[:3]
        skill_tabs = st.tabs([skill.title() for skill in top_skills])
        
        learning_resources = {
            "python": [
                "📚 **Python for Everybody** (Coursera - Free audit)",
                "📘 **Automate the Boring Stuff with Python** (Free book)",
                "🎥 **Corey Schafer Python Tutorials** (YouTube)",
                "💻 **LeetCode Python Challenges**"
            ],
            "machine learning": [
                "🏆 **Machine Learning by Andrew Ng** (Coursera)",
                "📚 **Hands-On Machine Learning with Scikit-Learn**",
                "🎓 **Fast.ai Practical Deep Learning**",
                "📊 **Kaggle Micro-courses**"
            ],
            "data analysis": [
                "📊 **Data Analysis with Python** (freeCodeCamp)",
                "📈 **Google Data Analytics Certificate** (Coursera)",
                "💼 **365 Data Science Data Analysis Course**",
                "📚 **Python for Data Analysis** by Wes McKinney"
            ],
            "react": [
                "⚛️ **React Official Tutorial** (reactjs.org)",
                "🎓 **Full Stack Open** (University of Helsinki)",
                "📚 **Epic React** by Kent C. Dodds",
                "💻 **Frontend Mentor React Challenges**"
            ],
            "aws": [
                "☁️ **AWS Cloud Practitioner Essentials** (Free)",
                "🏗️ **AWS Solutions Architect Associate** (Udemy)",
                "📚 **AWS Certified Solutions Architect Study Guide**",
                "💻 **AWS Free Tier Hands-on Labs**"
            ],
            "docker": [
                "🐳 **Docker Mastery** (Udemy)",
                "📚 **Docker Deep Dive** by Nigel Poulton",
                "🎥 **TechWorld with Nana Docker Course** (YouTube)",
                "💻 **Play with Docker** (Interactive Labs)"
            ]
        }
        
        for i, skill in enumerate(top_skills):
            with skill_tabs[i]:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #3B82F615 0%, #10B98115 100%);
                    border-radius: 15px;
                    padding: 1.5rem;
                    margin-bottom: 1rem;
                ">
                    <h4 style="color: #3B82F6; margin-top: 0;">Learning {skill.title()}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                if skill in learning_resources:
                    for resource in learning_resources[skill]:
                        st.markdown(f"""
                        <div style="
                            background: white;
                            border-radius: 10px;
                            padding: 1rem;
                            margin: 0.5rem 0;
                            border-left: 4px solid #3B82F6;
                        ">
                            {resource}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info(f"💡 **Suggested Search:** '{skill.title()} tutorial for beginners' on YouTube or Coursera")
                    
                    st.markdown("""
                    <div style="
                        background: #f8fafc;
                        border-radius: 10px;
                        padding: 1rem;
                        margin: 1rem 0;
                    ">
                        <h5>Quick Start Guide:</h5>
                        <ol>
                            <li>Watch beginner tutorials on YouTube</li>
                            <li>Practice on interactive platforms (DataCamp, freeCodeCamp)</li>
                            <li>Build a small project using the skill</li>
                            <li>Add it to your portfolio</li>
                        </ol>
                    </div>
                    """, unsafe_allow_html=True)
    
    # 30-Day Study Plan
    st.markdown("### 📅 30-Day Intensive Study Plan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #F59E0B15 0%, #EC489915 100%);
            border-radius: 15px;
            padding: 1.5rem;
            height: 100%;
        ">
            <h4 style="color: #F59E0B; margin-top: 0;">Week 1-2: Foundation</h4>
            <ul>
                <li>📚 Spend 1 hour daily on theoretical concepts</li>
                <li>🎬 Complete 2-3 beginner tutorials</li>
                <li>📝 Take notes and create cheatsheets</li>
                <li>💬 Join relevant online communities</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #10B98115 0%, #3B82F615 100%);
            border-radius: 15px;
            padding: 1.5rem;
            height: 100%;
        ">
            <h4 style="color: #10B981; margin-top: 0;">Week 3-4: Application</h4>
            <ul>
                <li>💻 Build a small project using the skill</li>
                <li>⚡ Practice daily with coding exercises</li>
                <li>👥 Participate in coding challenges</li>
                <li>📊 Add projects to your portfolio</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Practice Platforms
    st.markdown("### 💻 Recommended Practice Platforms")
    
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown("""
        <div style="
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            height: 100%;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        ">
            <h5 style="color: #3B82F6; margin-top: 0;">🎓 Free Learning</h5>
            <p><a href="https://www.kaggle.com/learn" target="_blank">• Kaggle Learn</a></p>
            <p><a href="https://www.freecodecamp.org/" target="_blank">• freeCodeCamp</a></p>
            <p><a href="https://www.datacamp.com/courses" target="_blank">• DataCamp Free</a></p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown("""
        <div style="
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            height: 100%;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        ">
            <h5 style="color: #10B981; margin-top: 0;">💻 Coding Practice</h5>
            <p><a href="https://leetcode.com/" target="_blank">• LeetCode</a></p>
            <p><a href="https://www.hackerrank.com/" target="_blank">• HackerRank</a></p>
            <p><a href="https://exercism.io/" target="_blank">• Exercism</a></p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown("""
        <div style="
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            height: 100%;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        ">
            <h5 style="color: #8B5CF6; margin-top: 0;">🏗️ Project Ideas</h5>
            <p><a href="https://madewithml.com/" target="_blank">• Made With ML</a></p>
            <p><a href="https://github.com/practical-tutorials/project-based-learning" target="_blank">• Project-Based Learning</a></p>
            <p><a href="https://github.com/ashishpatel26/500-AI-Machine-learning-Deep-learning-Computer-vision-NLP-Projects-with-code" target="_blank">• 500+ AI Projects</a></p>
        </div>
        """, unsafe_allow_html=True)

def admin_dashboard_mode(db_handler):
    """Admin dashboard mode with beautiful UI"""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
    ">
        <h2 style="color: white; margin: 0;">🔐 Admin Dashboard</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0;">
            Access analytics and manage user data
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Authentication
    col1, col2 = st.columns(2)
    
    with col1:
        ad_user = st.text_input("Username", placeholder="Enter admin username")
    
    with col2:
        ad_password = st.text_input("Password", type='password', placeholder="Enter admin password")
    
    if st.button('Login', type="primary", use_container_width=True):
        if ad_user == 'admin' and ad_password == 'admin123':
            st.success("✅ Welcome Admin!")
            
            # Fetch all data
            all_data = db_handler.get_all_analyses()
            
            if not all_data.empty:
                # Display data
                st.markdown("### 📊 All Analyses")
                st.dataframe(all_data, use_container_width=True)
                
                # Download link
                csv = all_data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="all_analyses.csv" style="text-decoration: none;">'
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #10B981 0%, #059669 100%);
                    color: white;
                    padding: 1rem;
                    border-radius: 10px;
                    text-align: center;
                    margin: 1rem 0;
                ">
                    {href}
                        <span style="font-size: 1.2rem;">📥 Download All Data</span>
                    </a>
                </div>
                """, unsafe_allow_html=True)
                
                # Analytics
                st.markdown("### 📈 Analytics Dashboard")
                
                # Field distribution
                if 'predicted_field' in all_data.columns:
                    fig1 = px.pie(all_data, names='predicted_field', 
                                title='Field Distribution',
                                color_discrete_sequence=px.colors.sequential.RdBu)
                    fig1.update_traces(textposition='inside', textinfo='percent+label')
                    fig1.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#374151'
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                # Experience distribution
                if 'total_experience' in all_data.columns:
                    fig2 = px.histogram(all_data, x='total_experience', 
                                       nbins=10, title='Experience Distribution',
                                       color_discrete_sequence=['#6366f1'])
                    fig2.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#374151',
                        xaxis_title="Years of Experience",
                        yaxis_title="Count"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No analysis data available yet.")
        else:
            st.error("❌ Invalid credentials")

def analyze_resume_mode(db_handler):
    """Resume analysis mode with beautiful UI - NO PREVIEW"""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
    ">
        <h2 style="color: white; margin: 0;">📄 Upload Your Resume for Analysis</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0;">
            Get detailed insights, ATS scores, and personalized recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload with beautiful design
    st.markdown("""
    <div style="
        background: white;
        border-radius: 20px;
        padding: 3rem;
        margin: 2rem auto;
        text-align: center;
        border: 3px dashed rgba(59, 130, 246, 0.3);
        max-width: 600px;
    ">
        <div style="font-size: 4rem; color: #3B82F6; margin-bottom: 1rem;">📄</div>
        <h3 style="color: #1F2937; margin-bottom: 0.5rem;">Drag & Drop Your Resume</h3>
        <p style="color: #6B7280; margin-bottom: 2rem;">Supports PDF, DOCX, and TXT formats</p>
    </div>
    """, unsafe_allow_html=True)
    
    pdf_file = st.file_uploader("Choose your Resume", type=["pdf", "docx", "txt"], label_visibility="collapsed")
    
    if pdf_file is not None:
        # Save uploaded file
        save_path = f'./Uploaded_Resumes/{pdf_file.name}'
        os.makedirs('./Uploaded_Resumes', exist_ok=True)
        
        with open(save_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        with st.spinner('🔍 Analyzing your resume with advanced NLP...'):
            # REMOVED THE TWO-COLUMN LAYOUT - NO PREVIEW NEEDED
            # Extract text
            resume_text = extract_text_from_file(save_path)
            
            if not resume_text or len(resume_text.strip()) < 50:
                st.error("Could not extract meaningful text from the resume.")
                return
            
            # Initialize and use the FIXED parser
            parser = EnhancedResumeParser()
            resume_data = parser.parse_resume(save_path)
            
            if not resume_data.get('name') and not resume_data.get('email'):
                st.warning("⚠️ Basic information not found. Check resume format.")
            
            # Display basic info with beautiful cards
            st.markdown("### 👤 Basic Information")
            
            info_cols = st.columns(3)
            with info_cols[0]:
                name = resume_data.get('name', 'Not found')
                st.markdown(create_metric_card(
                    "Name", 
                    name if name else "Not found", 
                    icon="👤", 
                    color="#3B82F6"
                ), unsafe_allow_html=True)
            
            with info_cols[1]:
                email = resume_data.get('email', 'Not found')
                st.markdown(create_metric_card(
                    "Email", 
                    email if email else "Not found", 
                    icon="📧", 
                    color="#10B981"
                ), unsafe_allow_html=True)
            
            with info_cols[2]:
                phone = resume_data.get('phone', 'Not found')
                st.markdown(create_metric_card(
                    "Phone", 
                    phone if phone else "Not found", 
                    icon="📱", 
                    color="#8B5CF6"
                ), unsafe_allow_html=True)
            
            # Rest of the code remains the same...
            # Advanced analysis tabs
            st.markdown("### 🤖 Advanced Analysis")
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Skills", 
                "🎯 Career", 
                "📈 ATS Score", 
                "🇮🇳 Indian Context",
                "📚 Education"
            ])
            
            with tab1:
                display_skills_analysis(resume_data)
            
            with tab2:
                display_career_path(resume_data)
            
            with tab3:
                display_ats_score(resume_text, resume_data)
            
            with tab4:
                display_enhanced_indian_context(resume_text, resume_data)
            
            with tab5:
                display_education_info(resume_data)
            
            # Recommendations
            display_recommendations(resume_data)
            
            # Store in database
            store_analysis_results(db_handler, resume_data)
            
            # Export options
            st.markdown("### 💾 Export Results")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Download JSON Report", use_container_width=True):
                    json_str = json.dumps(resume_data, indent=2, default=str)
                    st.download_button(
                        label="Click to Download",
                        data=json_str,
                        file_name=f"resume_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col2:
                if st.button("🖨️ Print Summary", use_container_width=True):
                    st.info("Print feature coming soon!")

def main():
    """Main application function with beautiful UI"""
    # Initialize database
    db_handler = DatabaseHandler()
    
    # Load custom CSS
    load_css()
    
    # Create beautiful header
    create_beautiful_header()
    
    # Create beautiful sidebar and get choice
    choice = create_beautiful_sidebar()
    
    # Route to selected mode
    if choice == "Analyze Resume":
        analyze_resume_mode(db_handler)
    elif choice == "Resume vs JD Matching":
        resume_jd_matching_mode()
    elif choice == "Skill Gap Analysis":
        skill_gap_analysis_mode()
    else:
        admin_dashboard_mode(db_handler)

if __name__ == "__main__":
    main()