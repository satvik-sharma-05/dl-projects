"""
Beautiful UI Components for Resume Analyzer
"""

import streamlit as st
from PIL import Image
import base64
import os

def load_css():
    """Load custom CSS"""
    css_file = os.path.join('static', 'css', 'style.css')
    if os.path.exists(css_file):
        with open(css_file, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        # Fallback inline CSS
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .main-header {
            background: white;
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)

def create_header():
    """Create beautiful header with logo and title"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Logo and Title
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <div style="display: inline-block; background: linear-gradient(45deg, #667eea, #764ba2); 
                     padding: 15px; border-radius: 50%; margin-bottom: 1rem;">
                <span style="font-size: 2.5rem;">🤖</span>
            </div>
            <h1 style="color: white; margin-bottom: 0.5rem; font-size: 2.5rem;">
                AI Resume Analyzer Pro
            </h1>
            <p style="color: rgba(255,255,255,0.8); font-size: 1.1rem;">
                Advanced NLP-powered resume analysis with Indian context support
            </p>
        </div>
        """, unsafe_allow_html=True)

def create_metric_card(title, value, icon="📊", color="#667eea", change=None):
    """Create a beautiful metric card"""
    if change:
        change_html = f'<div style="font-size: 0.9rem; color: {"#10B981" if change > 0 else "#EF4444"}">{"+" if change > 0 else ""}{change}%</div>'
    else:
        change_html = ""
    
    return f"""
    <div style="
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border-left: 5px solid {color};
        margin: 0.5rem;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <div style="
                background: {color}15;
                padding: 10px;
                border-radius: 10px;
                margin-right: 10px;
            ">
                <span style="font-size: 1.5rem;">{icon}</span>
            </div>
            <div style="font-size: 0.9rem; color: #6B7280;">{title}</div>
        </div>
        <div style="font-size: 2rem; font-weight: 700; color: #111827;">{value}</div>
        {change_html}
    </div>
    """

def create_progress_bar(label, value, color="#667eea"):
    """Create a styled progress bar"""
    return f"""
    <div style="margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="font-weight: 600; color: #374151;">{label}</span>
            <span style="font-weight: 700; color: {color};">{value}%</span>
        </div>
        <div style="
            width: 100%;
            height: 10px;
            background: #E5E7EB;
            border-radius: 10px;
            overflow: hidden;
        ">
            <div style="
                width: {value}%;
                height: 100%;
                background: linear-gradient(90deg, {color}, {color}80);
                border-radius: 10px;
                transition: width 1s ease-in-out;
            "></div>
        </div>
    </div>
    """

def create_skill_badge(skill, category="default"):
    """Create a colored skill badge"""
    category_colors = {
        "programming": "#3B82F6",
        "web": "#8B5CF6",
        "data": "#10B981",
        "cloud": "#F59E0B",
        "tool": "#EF4444",
        "soft": "#EC4899"
    }
    
    color = category_colors.get(category, "#6B7280")
    
    return f"""
    <span style="
        display: inline-block;
        background: {color}15;
        color: {color};
        padding: 4px 12px;
        border-radius: 20px;
        margin: 4px;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid {color}30;
    ">
        {skill}
    </span>
    """

def create_feature_card(icon, title, description):
    """Create a feature card for sidebar"""
    return f"""
    <div style="
        background: rgba(255,255,255,0.9);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    ">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-size: 1.2rem; margin-right: 10px;">{icon}</span>
            <span style="font-weight: 600; color: #111827;">{title}</span>
        </div>
        <div style="font-size: 0.9rem; color: #6B7280;">
            {description}
        </div>
    </div>
    """

def create_recommendation_card(type_, message, action, priority=1):
    """Create a recommendation card"""
    type_colors = {
        "critical": "#EF4444",
        "important": "#F59E0B",
        "experience": "#3B82F6",
        "category": "#8B5CF6",
        "overall": "#10B981"
    }
    
    type_icons = {
        "critical": "🚨",
        "important": "📝",
        "experience": "📈",
        "category": "🏷️",
        "overall": "🎯"
    }
    
    color = type_colors.get(type_.lower(), "#6B7280")
    icon = type_icons.get(type_.lower(), "💡")
    
    return f"""
    <div style="
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
        border-left: 4px solid {color};
        border-top: 1px solid #E5E7EB;
    ">
        <div style="display: flex; align-items: flex-start;">
            <div style="
                background: {color}15;
                padding: 8px;
                border-radius: 10px;
                margin-right: 12px;
            ">
                <span style="font-size: 1.2rem;">{icon}</span>
            </div>
            <div style="flex: 1;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0; color: #111827; font-weight: 600;">
                        {type_.title()}
                    </h4>
                    <span style="
                        background: {color};
                        color: white;
                        padding: 2px 8px;
                        border-radius: 12px;
                        font-size: 0.75rem;
                        font-weight: 600;
                    ">
                        Priority {priority}
                    </span>
                </div>
                <p style="margin: 0.5rem 0; color: #374151; font-size: 0.95rem;">
                    {message}
                </p>
                <div style="
                    background: #F9FAFB;
                    padding: 0.75rem;
                    border-radius: 8px;
                    margin-top: 0.5rem;
                    border-left: 3px solid {color};
                ">
                    <span style="font-weight: 600; color: {color};">💡 Action:</span>
                    <span style="color: #6B7280; margin-left: 5px;">{action}</span>
                </div>
            </div>
        </div>
    </div>
    """

def create_beautiful_header():
    """Create a beautiful modern header"""
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: center; gap: 20px; margin-bottom: 1rem;">
            <div style="
                background: rgba(255, 255, 255, 0.2);
                width: 80px;
                height: 80px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                backdrop-filter: blur(10px);
                border: 3px solid rgba(255, 255, 255, 0.3);
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            ">
                <span style="font-size: 2.5rem;">🤖</span>
            </div>
            <div style="text-align: left;">
                <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700; color: white;">
                    AI Resume Analyzer Pro
                </h1>
                <p style="margin: 0; font-size: 1.1rem; color: rgba(255, 255, 255, 0.9);">
                    Advanced NLP-powered resume analysis with Indian context support
                </p>
            </div>
        </div>
        <div style="
            display: inline-flex;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: center;
            margin-top: 1rem;
        ">
            <span style="
                background: rgba(255, 255, 255, 0.2);
                padding: 6px 15px;
                border-radius: 20px;
                font-size: 0.9rem;
                color: white;
                backdrop-filter: blur(10px);
            ">✅ Advanced NLP</span>
            <span style="
                background: rgba(255, 255, 255, 0.2);
                padding: 6px 15px;
                border-radius: 20px;
                font-size: 0.9rem;
                color: white;
                backdrop-filter: blur(10px);
            ">🇮🇳 Indian Context</span>
            <span style="
                background: rgba(255, 255, 255, 0.2);
                padding: 6px 15px;
                border-radius: 20px;
                font-size: 0.9rem;
                color: white;
                backdrop-filter: blur(10px);
            ">🎯 ATS Score</span>
            <span style="
                background: rgba(255, 255, 255, 0.2);
                padding: 6px 15px;
                border-radius: 20px;
                font-size: 0.9rem;
                color: white;
                backdrop-filter: blur(10px);
            ">🤖 ML Recommendations</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_beautiful_sidebar():
    """Create beautiful sidebar with enhanced styling"""
    with st.sidebar:
        # Sidebar header
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <div style="
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                width: 70px;
                height: 70px;
                border-radius: 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 1rem;
                box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3);
            ">
                <span style="font-size: 2rem; color: white;">📊</span>
            </div>
            <h3 style="color: #1f2937; margin: 0; font-weight: 600;">Resume Analyzer</h3>
            <p style="color: #6b7280; font-size: 0.9rem; margin: 0.25rem 0 0;">AI-Powered Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 🎯 Navigation")
        
        # Define the options with their icons
        options = [
            ("📄", "Analyze Resume"),
            ("📋", "Resume vs JD Matching"),
            ("🔧", "Skill Gap Analysis"),
            ("👨‍💼", "Admin Dashboard")
        ]
        
        # Extract just the text for the radio widget
        option_texts = [text for _, text in options]
        
        # Create a mapping for display
        display_map = {text: f"{icon} {text}" for icon, text in options}
        
        # Create custom radio buttons
        choice = st.radio(
            "Choose Mode:",
            option_texts,
            format_func=lambda x: display_map.get(x, x),
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Features
        st.markdown("### ✨ Features")
        
        features = [
            ("✅", "Advanced NLP Parsing", "Extract insights using state-of-the-art NLP models"),
            ("🇮🇳", "Indian Context", "Tailored analysis for Indian job market"),
            ("🎯", "ATS Optimization", "Improve your resume for Applicant Tracking Systems"),
            ("🤖", "AI Recommendations", "Get personalized career guidance"),
            ("📊", "Detailed Analytics", "Comprehensive skill and gap analysis")
        ]
        
        for icon, title, desc in features:
            with st.expander(f"{icon} {title}", expanded=False):
                st.caption(desc)
        
        st.markdown("---")
        
        # Stats
        st.markdown("### 📈 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Resumes Analyzed", "500+", "+25%")
        with col2:
            st.metric("Accuracy", "95%", "+2%")
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; color: #6b7280; font-size: 0.85rem;">
            <p style="margin: 0.5rem 0;">Made with ❤️ using Streamlit</p>
            <p style="margin: 0.5rem 0;">v2.1 • AI Resume Analyzer Pro</p>
            <p style="margin: 0.5rem 0; font-size: 0.8rem;">© 2024 All rights reserved</p>
        </div>
        """, unsafe_allow_html=True)
    
    return choice

def display_enhanced_metrics(match_results):
    """Display enhanced metrics with beautiful styling"""
    match_score = match_results.get('match_score', 0)
    
    # Determine color based on score
    if match_score >= 70:
        color = "#10B981"
        icon = "🎉"
        label = "Excellent Match"
    elif match_score >= 50:
        color = "#F59E0B"
        icon = "👍"
        label = "Good Match"
    else:
        color = "#EF4444"
        icon = "📈"
        label = "Needs Improvement"
    
    # Main score card
    st.markdown(f"""
    <div style="
        background: white;
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
        border: 2px solid {color}30;
    ">
        <div style="font-size: 3.5rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="font-size: 4rem; font-weight: 800; color: {color}; margin-bottom: 0.5rem;">
            {match_score:.1f}%
        </div>
        <div style="font-size: 1.3rem; color: #666; margin-bottom: 1.5rem; font-weight: 600;">
            {label}
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
                width: {match_score}%;
                height: 100%;
                background: linear-gradient(90deg, {color}, {color}80);
                border-radius: 10px;
                transition: width 1s ease-in-out;
            "></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Score breakdown in columns
    st.markdown("### 📊 Detailed Score Breakdown")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        skill_score = match_results.get('skill_match_score', 0)
        st.metric(
            label="Skill Match",
            value=f"{skill_score:.1f}%",
            delta="+" if skill_score >= 50 else "-"
        )
    
    with col2:
        semantic_score = match_results.get('semantic_score', 0)
        st.metric(
            label="Semantic Match",
            value=f"{semantic_score:.1f}%",
            delta="+" if semantic_score >= 50 else "-"
        )
    
    with col3:
        exp_score = match_results.get('experience_match_score', 0)
        st.metric(
            label="Experience Match",
            value=f"{exp_score:.1f}%",
            delta="+" if exp_score >= 50 else "-"
        )
    
    with col4:
        overall_score = match_results.get('match_score', 0)
        st.metric(
            label="Overall Score",
            value=f"{overall_score:.1f}%",
            delta="+" if overall_score >= 50 else "-"
        )

def create_skill_badges(skills, category_colors):
    """Create beautiful skill badges"""
    badge_html = ""
    for skill in skills:
        category = None
        skill_lower = skill.lower()
        
        # Determine category
        for cat, keywords in category_colors.items():
            if any(keyword in skill_lower for keyword in keywords):
                category = cat
                break
        
        color = category_colors.get(category, {"color": "#6B7280"})["color"]
        
        badge_html += f"""
        <span style="
            display: inline-block;
            background: {color}15;
            color: {color};
            padding: 6px 16px;
            border-radius: 20px;
            margin: 4px;
            font-size: 0.9rem;
            font-weight: 500;
            border: 1px solid {color}30;
            transition: all 0.3s ease;
        ">
            {skill.title()}
        </span>
        """
    
    return badge_html

def display_beautiful_skills_analysis(resume_data):
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
                    'git': '#F05032'
                })
                
                st.markdown(badge_html, unsafe_allow_html=True)
    else:
        st.info("No skills extracted. Try adding a 'SKILLS' section to your resume.")