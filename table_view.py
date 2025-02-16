import streamlit as st
import pandas as pd
import plotly.express as px
from unsupervised_analysis import analyze_patients_ml
import numpy as np
from datetime import datetime
import duckdb

def create_patient_table(data, patterns_df, priority_list):
    """Create a focused priority patient table"""
    
    # Debug: Print the first priority_info to see its structure
    if priority_list:
        print("Sample priority info:", priority_list[0])
    
    # Create base table dataframe
    table_data = []
    
    for patient_info in priority_list:
        patient = patient_info['patient']
        patient_pattern = patterns_df[patterns_df['patient'] == patient].iloc[0]
        
        # Calculate days since last visit from the data
        latest_visit = pd.to_datetime(data[data['patient'] == patient]['observation_date']).max()
        days_since_visit = (pd.Timestamp.now() - latest_visit).days
        
        # Get latest values from time series data
        latest_data = data[data['patient'] == patient].sort_values('observation_date').iloc[-1]
        
        # Format risk factors as a string
        risk_factors_str = '; '.join([
            f"{factor['factor']} ({factor['severity']})"
            for factor in patient_info.get('risk_factors', [])
        ])
        
        row = {
            'Patient ID': f"[{patient}](#)",  # Make patient ID clickable
            'Priority Score': f"{patient_info['priority_score']:.2f}",
            'Days Since Visit': days_since_visit,
            'Latest Stress': latest_data['stress_level'],
            'Social Contact': latest_data['social_contact_frequency'],
            'Housing Status': latest_data['housing_status'],
            'Employment': latest_data['employment_status'],
            'Risk Factors': risk_factors_str
        }
        table_data.append(row)
    
    # Create DataFrame
    table_df = pd.DataFrame(table_data)
    
    # Add styling
    def color_priority_score(val):
        try:
            score = float(val)
            if score > 0.7:
                return 'background-color: #ffcccc'
            elif score > 0.4:
                return 'background-color: #ffffcc'
            return 'background-color: #ccffcc'
        except:
            return ''

    # Style the dataframe
    styled_df = table_df.style.applymap(color_priority_score, subset=['Priority Score'])
    
    # Display metrics
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric(
            "Total Priority Patients", 
            len(table_df)
        )
        
    with metrics_col2:
        high_priority = len(table_df[pd.to_numeric(table_df['Priority Score']) > 0.7])
        st.metric(
            "High Priority Patients",
            high_priority,
            delta=f"{(high_priority/len(table_df)*100):.1f}% of total"
        )
        
    with metrics_col3:
        avg_days = table_df['Days Since Visit'].mean()
        st.metric(
            "Avg Days Since Visit",
            f"{avg_days:.1f}"
        )
    
    # Display table
    st.dataframe(
        table_df.sort_values('Priority Score', ascending=False),
        height=400,
        use_container_width=True
    )
    
    return table_df

def main():
    # Get data from unsupervised analysis
    patterns_df, insights, priority_list = analyze_patients_ml()
    
    # Get the data from the same source
    @st.cache_data
    def get_cached_data():
        conn = duckdb.connect()
        
        # Register CSV files
        conn.execute("CREATE TABLE csv_data_observations AS SELECT * FROM read_csv_auto('csv/observations.csv')")
        conn.execute("CREATE TABLE csv_data_conditions AS SELECT * FROM read_csv_auto('csv/conditions.csv')")
        
        # Query to get social determinants data
        query = """
        WITH patient_timeline AS (
            SELECT 
                o.patient,
                o.date as observation_date,
                MAX(CASE WHEN o.description = 'Stress level' THEN o.value END) as stress_level,
                MAX(CASE WHEN o.description = 'Employment status - current' THEN o.value END) as employment_status,
                MAX(CASE WHEN o.description = 'Housing status' THEN o.value END) as housing_status,
                MAX(CASE WHEN o.description = 'Are you worried about losing your housing?' THEN o.value END) as housing_worry,
                MAX(CASE WHEN LOWER(o.description) LIKE '%see or talk to people%' THEN o.value END) as social_contact_frequency
            FROM csv_data_observations o
            WHERE o.type = 'text'
            AND (
                o.description = 'Stress level'
                OR o.description = 'Employment status - current'
                OR o.description = 'Housing status'
                OR o.description = 'Are you worried about losing your housing?'
                OR LOWER(o.description) LIKE '%see or talk to people%'
            )
            GROUP BY o.patient, o.date
            HAVING COUNT(DISTINCT o.description) >= 3
        )
        SELECT * FROM patient_timeline
        ORDER BY patient, observation_date;
        """
        
        df = conn.execute(query).df()
        conn.close()
        return df

    data = get_cached_data()
    
    # Create table view
    filtered_df = create_patient_table(data, patterns_df, priority_list)
    
    # Add visualization of selected patients
    st.subheader("Selected Patients Distribution")
    
    fig = px.scatter(
        patterns_df[patterns_df['patient'].isin(filtered_df['Patient ID'].str.strip('[]#'))],
        x='pca_x',
        y='pca_y',
        color='cluster',
        title='Patient Clusters (PCA)',
        hover_data=['patient', 'mean_stress', 'mean_social_contact']
    )
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()