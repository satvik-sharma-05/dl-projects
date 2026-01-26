"""
Visualization utilities for resume analysis
"""

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import base64
from io import BytesIO
from typing import Dict, List, Any
import json

def create_resume_visualizations(resume_data: Dict, 
                                prediction_results: Dict) -> go.Figure:
    """Create comprehensive visualizations for resume analysis"""
    # Create subplots
    fig = go.Figure()
    
    # 1. Skills word cloud (if skills exist)
    if 'skills' in resume_data and resume_data['skills']:
        skills_text = ' '.join(resume_data['skills'])
        create_skills_wordcloud(skills_text)
    
    # 2. Field prediction probabilities
    if 'probabilities' in prediction_results:
        fig = create_prediction_chart(prediction_results['probabilities'])
    
    # 3. Experience timeline (if dates extracted)
    if 'dates' in resume_data:
        fig = create_experience_timeline(resume_data['dates'])
    
    return fig

def create_skills_wordcloud(skills_text: str):
    """Create word cloud from skills"""
    if not skills_text:
        return None
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=50
    ).generate(skills_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    
    # Convert to base64 for displaying in Streamlit
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode()

def create_prediction_chart(probabilities: Dict[str, float]) -> go.Figure:
    """Create bar chart for field prediction probabilities"""
    df = pd.DataFrame({
        'Field': list(probabilities.keys()),
        'Probability': list(probabilities.values())
    }).sort_values('Probability', ascending=True)
    
    fig = px.bar(
        df,
        y='Field',
        x='Probability',
        orientation='h',
        title='Field Prediction Probabilities',
        color='Probability',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title="Probability",
        yaxis_title="Field",
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_experience_timeline(dates: List[str]) -> go.Figure:
    """Create experience timeline visualization"""
    # Parse dates and create timeline
    # This is a simplified version
    timeline_data = []
    
    for i, date_str in enumerate(dates[:5]):  # Limit to 5 dates
        timeline_data.append({
            'Position': f'Role {i+1}',
            'Start': f'20{20-i}-01',
            'End': f'20{21-i}-12' if i > 0 else 'Present',
            'Duration': f'{i+1} year(s)'
        })
    
    df = pd.DataFrame(timeline_data)
    
    fig = px.timeline(
        df,
        x_start="Start",
        x_end="End",
        y="Position",
        color="Duration",
        title="Experience Timeline",
        hover_data=["Duration"]
    )
    
    fig.update_layout(
        height=300,
        template='plotly_white'
    )
    
    return fig

def create_skill_gap_chart(current_skills: List[str], 
                          required_skills: List[str]) -> go.Figure:
    """Create skill gap analysis chart"""
    current_set = set(current_skills)
    required_set = set(required_skills)
    
    matched = current_set.intersection(required_set)
    missing = required_set - current_set
    extra = current_set - required_set
    
    data = {
        'Category': ['Matched', 'Missing', 'Extra'],
        'Count': [len(matched), len(missing), len(extra)]
    }
    
    df = pd.DataFrame(data)
    
    fig = px.pie(
        df,
        values='Count',
        names='Category',
        title='Skill Gap Analysis',
        color='Category',
        color_discrete_map={
            'Matched': '#00cc96',
            'Missing': '#ef553b',
            'Extra': '#636efa'
        }
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig

def create_salary_comparison_chart(role: str, 
                                 experience: int, 
                                 location: str,
                                 salary_data: Dict) -> go.Figure:
    """Create salary comparison chart"""
    # Sample salary data structure
    # salary_data = {
    #     'Bangalore': [6, 12, 18, 25],
    #     'Delhi': [5, 10, 16, 22],
    #     'Mumbai': [6, 11, 17, 24]
    # }
    
    levels = ['Fresher', 'Junior', 'Mid', 'Senior']
    
    fig = go.Figure()
    
    for city, salaries in salary_data.items():
        fig.add_trace(go.Scatter(
            x=levels,
            y=salaries,
            name=city,
            mode='lines+markers'
        ))
    
    # Add marker for user's level
    if experience < 1:
        user_level = 'Fresher'
    elif experience < 3:
        user_level = 'Junior'
    elif experience < 6:
        user_level = 'Mid'
    else:
        user_level = 'Senior'
    
    # Find salary for user's level in their location
    if location in salary_data:
        user_salary = salary_data[location][levels.index(user_level)]
        fig.add_trace(go.Scatter(
            x=[user_level],
            y=[user_salary],
            mode='markers',
            marker=dict(size=15, color='red'),
            name='Your Level'
        ))
    
    fig.update_layout(
        title=f'Salary Comparison for {role}',
        xaxis_title='Experience Level',
        yaxis_title='Salary (LPA)',
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_skill_trend_chart(skill_data: pd.DataFrame) -> go.Figure:
    """Create skill trend chart over time"""
    fig = px.line(
        skill_data,
        x='Year',
        y='Demand',
        color='Skill',
        title='Skill Demand Trends',
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Demand Score",
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_resume_score_radar(score_breakdown: Dict[str, float]) -> go.Figure:
    """Create radar chart for resume score breakdown"""
    categories = list(score_breakdown.keys())
    scores = list(score_breakdown.values())
    
    fig = go.Figure(data=go.Scatterpolar(
        r=scores + [scores[0]],  # Close the radar
        theta=categories + [categories[0]],
        fill='toself',
        name='Resume Score'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title='Resume Score Breakdown',
        height=400
    )
    
    return fig

def create_industry_demand_chart(demand_data: pd.DataFrame) -> go.Figure:
    """Create industry demand chart"""
    fig = px.bar(
        demand_data,
        x='Industry',
        y='Demand',
        color='Demand',
        title='Industry Demand Trends',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title="Industry",
        yaxis_title="Demand Score",
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_geographic_heatmap(location_data: pd.DataFrame) -> go.Figure:
    """Create geographic heatmap for job locations"""
    # This would typically use actual geographic data
    # Simplified version with Indian cities
    
    fig = px.scatter_geo(
        location_data,
        lat='lat',
        lon='lon',
        size='job_count',
        color='average_salary',
        hover_name='city',
        title='Job Distribution Across India',
        scope='asia',
        projection='natural earth',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=500,
        geo=dict(
            showland=True,
            landcolor="rgb(243, 243, 243)",
            countrycolor="rgb(204, 204, 204)",
            lataxis_range=[8, 38],  # India latitude range
            lonaxis_range=[68, 98]   # India longitude range
        )
    )
    
    return fig

def create_learning_path_timeline(skills_to_learn: List[str]) -> go.Figure:
    """Create learning path timeline"""
    timeline_data = []
    
    for i, skill in enumerate(skills_to_learn[:6]):  # Limit to 6 skills
        timeline_data.append({
            'Skill': skill,
            'Start Week': i * 4,
            'End Week': (i + 1) * 4,
            'Resources': f'{i+1} courses'
        })
    
    df = pd.DataFrame(timeline_data)
    
    fig = px.timeline(
        df,
        x_start="Start Week",
        x_end="End Week",
        y="Skill",
        color="Skill",
        title="Recommended Learning Path",
        hover_data=["Resources"]
    )
    
    fig.update_layout(
        xaxis_title="Weeks",
        yaxis_title="Skills",
        height=300,
        template='plotly_white'
    )
    
    return fig