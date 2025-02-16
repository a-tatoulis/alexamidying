import streamlit as st
import pandas as pd
import duckdb
from pattern_detection import PatternDetector
import plotly.graph_objects as go
import plotly.express as px
from table_view import create_patient_table
from datetime import datetime
from unsupervised_analysis import analyze_patients_ml
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.grid_options_builder import GridOptionsBuilder
import streamlit.components.v1 as components

# Set page config with dark theme
st.set_page_config(
    page_title="Patient Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_data():
    try:
        conn = duckdb.connect()
        
        # Register all needed CSV files with caching
        conn.execute("""
            CREATE TABLE IF NOT EXISTS csv_data_observations AS 
            SELECT * FROM read_csv_auto('csv/observations.csv')
            WHERE type in ('text', 'numeric')
            AND description IN (
                'Stress level',
                'Body Weight',
                'Employment status - current',
                'Housing status',
                'Are you worried about losing your housing?'
            ) OR LOWER(description) LIKE '%see or talk to people%'
        """)
        
        conn.execute("CREATE TABLE IF NOT EXISTS csv_data_conditions AS SELECT * FROM read_csv_auto('csv/conditions.csv') WHERE stop IS NULL")
        conn.execute("CREATE TABLE IF NOT EXISTS csv_data_patients AS SELECT Id, FIRST, LAST, BIRTHDATE, GENDER, RACE, ETHNICITY, CITY, STATE FROM read_csv_auto('csv/patients.csv')")
        
        # Modified query with correct column names
        query = """
        WITH RECURSIVE patient_timeline AS (
            SELECT 
                o.PATIENT,
                o.DATE as observation_date,
                MAX(CASE WHEN o.description = 'Stress level' THEN o.value END) as stress_level,
                MAX(CASE WHEN o.description = 'Body Weight' THEN CAST(o.value AS FLOAT) END) as weight,
                MAX(CASE WHEN o.description = 'Employment status - current' THEN o.value END) as employment_status,
                MAX(CASE WHEN o.description = 'Housing status' THEN o.value END) as housing_status,
                MAX(CASE WHEN o.description = 'Are you worried about losing your housing?' THEN o.value END) as housing_worry,
                MAX(CASE WHEN LOWER(o.description) LIKE '%see or talk to people%' THEN o.value END) as social_contact_frequency,
                MIN(o.DATE) OVER (PARTITION BY o.PATIENT) as diagnosis_date,
                DATEDIFF('day', MIN(o.DATE) OVER (PARTITION BY o.PATIENT), o.DATE) as days_since_diagnosis
            FROM csv_data_observations o
            GROUP BY o.PATIENT, o.DATE
            HAVING COUNT(DISTINCT o.description) >= 3
        ),
        current_conditions AS (
            SELECT 
                patient,
                STRING_AGG(DISTINCT description, '; ') as current_conditions
            FROM csv_data_conditions
            GROUP BY patient
        )
        SELECT 
            pt.*,
            p.FIRST as first_name,
            p.LAST as last_name,
            p.BIRTHDATE,
            p.GENDER,
            p.RACE,
            p.ETHNICITY,
            p.CITY,
            p.STATE,
            cc.current_conditions
        FROM patient_timeline pt
        LEFT JOIN csv_data_patients p ON pt.PATIENT = p.Id
        LEFT JOIN current_conditions cc ON pt.PATIENT = cc.patient
        ORDER BY pt.PATIENT, pt.observation_date;
        """
        
        # Execute query with a timeout
        df = conn.execute(query).df()
        
        # Add derived columns using vectorized operations
        stress_map = {
            'Not at all': 1,
            'A little bit': 2,
            'Somewhat': 3,
            'Quite a bit': 4,
            'Very much': 5
        }
        df['stress_level_numeric'] = pd.Categorical(df['stress_level']).map(stress_map)
        
        contact_map = {
            'Less than once a week': 1,
            '1 or 2 times a week': 2,
            '3 to 5 times a week': 3,
            '5 or more times a week': 4
        }
        df['social_contact_numeric'] = pd.Categorical(df['social_contact_frequency']).map(contact_map)
        
        # Convert dates efficiently
        df['observation_date'] = pd.to_datetime(df['observation_date'])
        
        conn.close()
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def get_cached_data():
    return load_data()

def create_gauge_chart(value, title, trend=None):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'color': 'white'}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickfont': {'color': 'white'}},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "lightyellow"},
                {'range': [70, 100], 'color': "lightpink"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 2},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    # Add a trend arrow if provided
    if trend is not None:
        arrow = '↑' if trend.lower() == 'up' else '↓' if trend.lower() == 'down' else ''
        fig.add_annotation(x=0.5, y=0.1, text=arrow, showarrow=False, font=dict(size=20, color="white"))
    fig.update_layout(
        height=200,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    return fig

def display_patient_details(data, patient_id, pattern, recommendations, patient_features):
    # Get patient's latest data
    patient_data = data[data['PATIENT'] == patient_id].iloc[-1]
    patient_history = data[data['PATIENT'] == patient_id].sort_values('observation_date')
    
    # Calculate age
    birth_date = pd.to_datetime(patient_data['BIRTHDATE'])
    age = (pd.Timestamp.now() - birth_date).days // 365

    # Create header with status indicator
    st.markdown("### Patient Overview")
    status_col, name_col = st.columns([1, 6])
    with status_col:
        if pattern.get('priority_score', 0) > 0.7:
            st.error("⚠️ High Risk")
        else:
            st.success("✓ Stable")
    with name_col:
        st.subheader(f"{patient_data['first_name']} {patient_data['last_name']}")
    
    # Demographics in columns with tooltips
    demo_col1, demo_col2, demo_col3 = st.columns(3)
    with demo_col1:
        st.metric("Age", age)
        st.metric("Gender", patient_data['GENDER'])
    with demo_col2:
        st.metric("Race", patient_data['RACE'])
        st.metric("Ethnicity", patient_data['ETHNICITY'])
    with demo_col3:
        st.metric("Location", f"{patient_data['CITY']}, {patient_data['STATE']}")
    
    # Medical conditions in expander
    with st.expander("Medical Conditions", expanded=False):
        if patient_data['current_conditions']:
            conditions = patient_data['current_conditions'].split('; ')
            disorders = [c.replace('(disorder)', '').strip() for c in conditions if '(disorder)' in c.lower()]
            findings = [c.replace('(finding)', '').strip() for c in conditions if '(finding)' in c.lower()]
            
            if disorders:
                st.markdown("**Disorders:**")
                for disorder in disorders:
                    st.info(disorder)
            if findings:
                st.markdown("**Findings:**")
                for finding in findings:
                    st.warning(finding)
        else:
            st.info("No conditions recorded")
    
    # Main dashboard content
    metrics_col, details_col = st.columns([3, 2])
    
    with metrics_col:
        # Weight management metrics
        st.markdown("#### Weight Management Progress")
        
        # Calculate weight change metrics
        if 'weight' in patient_history.columns:
            weight_data = patient_history[patient_history['weight'].notna()]
            if len(weight_data) >= 2:
                initial_weight = weight_data['weight'].iloc[0]
                current_weight = weight_data['weight'].iloc[-1]
                total_change = current_weight - initial_weight
                pct_change = (total_change / initial_weight) * 100
                
                # Create weight change metrics
                weight_col1, weight_col2 = st.columns(2)
                with weight_col1:
                    st.metric(
                        "Total Weight Change",
                        f"{total_change:.1f} lbs",
                        delta=f"{pct_change:.1f}%",
                        delta_color="inverse"
                    )
                with weight_col2:
                    st.metric(
                        "Current Weight",
                        f"{current_weight:.1f} lbs"
                    )
                
                # Weight trend chart
                st.markdown("##### Weight Trend")
                weight_chart = alt.Chart(weight_data).mark_line(point=True).encode(
                    x='observation_date:T',
                    y=alt.Y('weight:Q', scale=alt.Scale(zero=False)),
                    tooltip=['observation_date:T', 'weight:Q']
                ).properties(height=200)
                st.altair_chart(weight_chart, use_container_width=True)
        
        # Medication adherence
        st.markdown("#### Medication Adherence")
        try:
            if 'prescription_count' in patient_data and pd.notnull(patient_data['prescription_count']) and patient_data['prescription_count'] > 0:
                med_col1, med_col2 = st.columns(2)
                with med_col1:
                    st.metric(
                        "Active Prescriptions",
                        patient_data.get('active_prescriptions', 0)
                    )
                with med_col2:
                    st.metric(
                        "Total Dispenses",
                        patient_data.get('total_dispenses', 0)
                    )
            else:
                st.info("No GLP-1 medication history")
        except Exception as e:
            st.info("No GLP-1 medication history")
            print(f"Error displaying medication adherence: {str(e)}")
        
        # Visit adherence
        st.markdown("#### Visit Adherence")
        visit_col1, visit_col2 = st.columns(2)
        with visit_col1:
            st.metric(
                "Total Visits",
                patient_data['total_visits']
            )
        with visit_col2:
            st.metric(
                "Days Since Last Visit",
                patient_data['days_since_last_visit']
            )
        
        # Progress bars
        st.markdown("#### Progress Metrics")
        
        # Weight loss progress
        if 'weight' in patient_history.columns and len(weight_data) >= 2:
            weight_progress = min(abs(pct_change) / 5, 1.0)  # Target 5% weight loss
            st.progress(weight_progress, text=f"Weight Loss Progress: {weight_progress*100:.0f}%")
        
        # Medication adherence progress
        if pd.notnull(patient_data['prescription_count']) and patient_data['prescription_count'] > 0:
            med_progress = patient_data['active_prescriptions'] / patient_data['prescription_count']
            st.progress(med_progress, text=f"Medication Adherence: {med_progress*100:.0f}%")
        
        # Visit adherence progress
        if pd.notnull(patient_data['days_since_last_visit']) and patient_data['days_since_last_visit'] <= 30:
            visit_progress = max(0, 1 - (patient_data['days_since_last_visit'] / 90))  # Target: visit within 90 days
            st.progress(visit_progress, text=f"Visit Adherence: {visit_progress*100:.0f}%")
    
    with details_col:
        # Risk factors section
        st.markdown("#### Risk Factors")
        for factor in pattern.get('risk_factors', []):
            severity = factor['severity']
            # Get human readable description
            description = ""
            if factor['factor'] == 'Housing Concerns':
                description = "Consistent unstable housing situation"
            elif factor['factor'] == 'Employment Instability':
                description = f"{factor['value']} job changes in the past year"
            elif factor['factor'] == 'Extended Gap in Care':
                description = f"{int(factor['value'])} days since last visit"
            elif factor['factor'] == 'High Stress Levels':
                description = f"Stress level {factor['value']:.1f} out of 5"
            elif factor['factor'] == 'Low Social Contact':
                description = f"Only {factor['value']:.1f} social interactions per week"
            else:
                description = f"Value: {factor['value']:.1f}"

            if severity == 'high':
                st.error(f"⚠️ High Risk: {factor['factor']}\n{description}")
            elif severity == 'medium':
                st.warning(f"⚡ Medium Risk: {factor['factor']}\n{description}")
            else:
                st.info(f"ℹ️ Low Risk: {factor['factor']}\n{description}")
        
        # Positive factors section
        st.markdown("#### Positive Factors")
        if 'weight' in patient_history.columns and len(weight_data) >= 2:
            if pct_change <= -5:
                st.success("**Significant Weight Loss** 🌟", help=f"Lost {abs(pct_change):.1f}% of initial weight")
            elif pct_change <= -2:
                st.success("**Moderate Weight Loss** ⭐", help=f"Lost {abs(pct_change):.1f}% of initial weight")
            elif pct_change < 0:
                st.success("**Initial Weight Loss** 👍", help=f"Lost {abs(pct_change):.1f}% of initial weight")
        
        if pd.notnull(patient_data['prescription_count']) and patient_data['active_prescriptions'] > 0:
            st.success("**Active Medication Adherence** 💊", help="Currently taking prescribed GLP-1 medication")
        
        if pd.notnull(patient_data['days_since_last_visit']) and patient_data['days_since_last_visit'] <= 30:
            st.success("**Regular Visit Attendance** ✅", help="Visited within the last 30 days")
        
        # Trends section
        if pattern.get('trends', []):
            st.markdown("#### Trends")
            for trend in pattern['trends']:
                st.markdown(f"📈 {trend}")
        
        # Display recommendations
        st.markdown("#### Recommendations")
        for rec in pattern.get('recommendations', []):
            st.success(f"**{rec}**")

def create_aggrid_table(data, priority_list):
    """Create a DataFrame formatted for AgGrid display"""
    # Create base DataFrame
    priority_df = pd.DataFrame(priority_list)
    
    # Get latest records
    latest_records = data.sort_values('observation_date').groupby('PATIENT').last().reset_index()
    
    # Merge data
    full_table = latest_records.merge(
        priority_df[['patient', 'priority_score', 'risk_factors', 'recommendations']],
        left_on='PATIENT',
        right_on='patient',
        how='left'
    )
    
    # Format data
    display_df = pd.DataFrame({
        'First Name': full_table['first_name'],
        'Last Name': full_table['last_name'],
        'Risk Score': (full_table['priority_score'] * 100).round().astype(int),
        'Risk Factors': full_table['risk_factors'].apply(
            lambda x: '; '.join([f"{f['factor']} ({f['severity']})" for f in x]) if isinstance(x, list) else ''
        ),
        'Recommendations': full_table['recommendations'].apply(
            lambda x: '; '.join(x) if isinstance(x, list) else ''
        ),
        'State': full_table['STATE'],
        'Days Since Visit': (datetime.now() - pd.to_datetime(full_table['observation_date'])).dt.days.fillna(0).astype(int)
    })
    
    return display_df.sort_values('Risk Score', ascending=False)

def main():
    # Data source status using st.status
    with st.status("Data Source Status", expanded=True) as status:
        data = get_cached_data()
        if data is not None and not data.empty:
            status.update(label="Data Source: Connected", state="complete", expanded=False)
        else:
            status.update(label="Data Source: Error", state="error")
            st.error("Failed to load data. Please check the data file.")
            return

    # Dashboard title with metrics
    st.title("Patient Risk Dashboard")
    
    # Get analysis data
    patterns_df, insights, priority_list = get_cached_analysis()
    weight_metrics, adherence_metrics, attendance_metrics, progress_scores = get_cached_weight_analysis()
    
    # Calculate summary metrics
    total_patients = len(priority_list)
    high_risk = sum(1 for p in priority_list if p['priority_score'] > 0.7)
    med_risk = sum(1 for p in priority_list if 0.4 < p['priority_score'] <= 0.7)
    
    # Calculate weight management metrics
    total_weight_patients = len(progress_scores)
    high_progress = sum(1 for p in progress_scores if p['progress_score'] > 0.7)
    med_progress = sum(1 for p in progress_scores if 0.4 <= p['progress_score'] <= 0.7)
    
    # Display key metrics in two rows
    st.markdown("### Risk Assessment")
    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
    with risk_col1:
        st.metric(label="Total Patients", value=total_patients)
    with risk_col2:
        st.metric(label="High Risk", value=high_risk, delta=f"{(high_risk/total_patients)*100:.1f}%")
    with risk_col3:
        st.metric(label="Medium Risk", value=med_risk, delta=f"{(med_risk/total_patients)*100:.1f}%")
    with risk_col4:
        st.metric(label="Active Interventions", value=len([p for p in priority_list if p.get('recommendations')]))
    
    st.markdown("### Weight Management")
    weight_col1, weight_col2, weight_col3, weight_col4 = st.columns(4)
    with weight_col1:
        st.metric(label="Obese Patients", value=total_weight_patients)
    with weight_col2:
        st.metric(label="High Progress", value=high_progress, delta=f"{(high_progress/total_weight_patients)*100:.1f}%")
    with weight_col3:
        st.metric(label="Medium Progress", value=med_progress, delta=f"{(med_progress/total_weight_patients)*100:.1f}%")
    with weight_col4:
        active_meds = sum(1 for p in progress_scores if p['medication_metrics']['active_prescription'])
        st.metric(label="Active GLP-1 Rx", value=active_meds)

    # Create tabs for different views
    tab1, tab2 = st.tabs(["Priority Patients", "All Patients"])
    
    with tab1:
        # Add CSS for patient cards
        st.markdown("""
            <style>
            .patient-card {
                border: 1px solid #444;
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
                background-color: #1E1E1E;
            }
            .patient-header {
                padding-bottom: 10px;
                margin-bottom: 15px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .patient-name {
                font-size: 1.5em;
                font-weight: bold;
                color: white;
            }
            .risk-badge {
                padding: 5px 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            .high-risk {
                background-color: #ff4444;
                color: white;
            }
            .medium-risk {
                background-color: #ffa726;
                color: black;
            }
            .low-risk {
                background-color: #66bb6a;
                color: black;
            }
            .medical-item {
                background-color: #333;
                padding: 8px;
                margin: 4px 0;
                border-radius: 4px;
                color: #ccc;
            }
            .medical-item:hover .additional-conditions {
                display: block;
            }
            .additional-conditions {
                display: none;
                margin-top: 8px;
                padding-top: 8px;
                border-top: 1px solid #444;
            }
            .action-button {
                display: flex;
                align-items: center;
                justify-content: space-between;
                background-color: #ffa726;
                color: black;
                padding: 10px 15px;
                border-radius: 5px;
                margin: 5px 0;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            .action-button:hover {
                background-color: #ffb74d;
            }
            .action-button.completed {
                background-color: #66bb6a;
                color: white;
            }
            .action-button .cta {
                font-weight: bold;
                margin-left: 10px;
            }
            .action-button .arrow {
                margin-left: auto;
            }
            </style>
        """, unsafe_allow_html=True)

        # Merge priority list with weight management data
        priority_with_weight = []
        for patient in priority_list:
            # Find matching weight data
            weight_score = next(
                (score for score in progress_scores if score['PATIENT'] == patient['patient']),
                None
            )
            if weight_score:  # Only include patients with weight data
                patient['weight_data'] = weight_score
                priority_with_weight.append(patient)
        
        # Sort by priority score and get top 5
        top_priority = sorted(
            priority_with_weight,
            key=lambda x: x['priority_score'],
            reverse=True
        )[:5]

        st.markdown("### High Risk Patients with Weight Management Data")
        
        # Display priority patients with improved layout
        for i, patient in enumerate(top_priority):
            try:
                patient_data = data[data['PATIENT'] == patient['patient']].iloc[-1]
                patient_history = data[data['PATIENT'] == patient['patient']].sort_values('observation_date')
                
                # Calculate age
                birth_date = pd.to_datetime(patient_data['BIRTHDATE'])
                age = (pd.Timestamp.now() - birth_date).days // 365
                
                # Create patient card using Streamlit container
                with st.container():
                    # Header section with name and risk level
                    header_cols = st.columns([4, 1])
                    with header_cols[0]:
                        st.subheader(f"🏥 {patient_data['first_name']} {patient_data['last_name']}")
                    with header_cols[1]:
                        if patient.get('priority_score', 0) > 0.7:
                            st.error("High Risk")
                        elif patient.get('priority_score', 0) > 0.4:
                            st.warning("Medium Risk")
                        else:
                            st.success("Low Risk")
                    
                    # Patient details in an expander
                    with st.expander("View Patient Details", expanded=False):
                        # Demographics section
                        st.markdown("#### Demographics")
                        demo_cols = st.columns(4)
                        with demo_cols[0]:
                            st.metric("Age", age)
                        with demo_cols[1]:
                            st.metric("Gender", patient_data['GENDER'])
                        with demo_cols[2]:
                            st.metric("Race", patient_data['RACE'])
                        with demo_cols[3]:
                            st.metric("Location", f"{patient_data['CITY']}, {patient_data['STATE']}")
                        
                        st.divider()
                        
                        # Main content section
                        main_cols = st.columns(4)
                        
                        # Risk Factors column
                        with main_cols[0]:
                            st.markdown("##### Risk Factors")
                            for factor in patient.get('risk_factors', []):
                                severity = factor['severity']
                                # Get human readable description
                                description = ""
                                if factor['factor'] == 'Housing Concerns':
                                    description = "Consistent unstable housing situation"
                                elif factor['factor'] == 'Employment Instability':
                                    description = f"{factor['value']} job changes in the past year"
                                elif factor['factor'] == 'Extended Gap in Care':
                                    description = f"{int(factor['value'])} days since last visit"
                                elif factor['factor'] == 'High Stress Levels':
                                    description = f"Stress level {factor['value']:.1f} out of 5"
                                elif factor['factor'] == 'Low Social Contact':
                                    description = f"Only {factor['value']:.1f} social interactions per week"
                                else:
                                    description = f"Value: {factor['value']:.1f}"

                                if severity == 'high':
                                    st.error(f"⚠️ {factor['factor']}\n{description}")
                                elif severity == 'medium':
                                    st.warning(f"⚡ {factor['factor']}\n{description}")
                                else:
                                    st.info(f"ℹ️ {factor['factor']}\n{description}")
                        
                        # Weight Management column
                        with main_cols[1]:
                            st.markdown("##### Weight Management")
                            weight_data = patient['weight_data']
                            weight_metrics = weight_data['weight_metrics']
                            med_metrics = weight_data['medication_metrics']
                            
                            if med_metrics['active_prescription']:
                                st.success("✅ Active GLP-1 Medication")
                                st.metric(
                                    "Days on Medication",
                                    med_metrics['days_on_medication']
                                )
                            else:
                                st.warning("⚠️ No Active GLP-1 Medication")
                            
                            st.metric(
                                "Total Weight Change",
                                f"{weight_metrics['total_change']:.1f} lbs",
                                f"{weight_metrics['pct_change']:.1f}%",
                                delta_color="inverse"
                            )
                        
                        # Medical Conditions column
                        with main_cols[2]:
                            st.markdown("##### Medical Conditions")
                            if patient_data['current_conditions']:
                                conditions = patient_data['current_conditions'].split('; ')
                                disorders = [c.replace('(disorder)', '').strip() for c in conditions if '(disorder)' in c.lower()]
                                
                                if disorders:
                                    visible_disorders = disorders[:3]
                                    hidden_disorders = disorders[3:]
                                    
                                    # Display visible disorders
                                    for disorder in visible_disorders:
                                        with st.container():
                                            st.markdown(f"🏥 {disorder}")
                                    
                                    # Show hidden disorders with a toggle
                                    if hidden_disorders:
                                        show_more = st.button(
                                            f"+ {len(hidden_disorders)} more disorders",
                                            key=f"disorders_{i}"
                                        )
                                        if show_more:
                                            for disorder in hidden_disorders:
                                                st.markdown(f"🏥 {disorder}")
                            else:
                                st.info("No conditions recorded")
                        
                        # Clinical Findings column
                        with main_cols[3]:
                            st.markdown("##### Clinical Findings")
                            if patient_data['current_conditions']:
                                conditions = patient_data['current_conditions'].split('; ')
                                findings = [c.replace('(finding)', '').strip() for c in conditions if '(finding)' in c.lower()]
                                
                                if findings:
                                    visible_findings = findings[:3]
                                    hidden_findings = findings[3:]
                                    
                                    # Display visible findings
                                    for finding in visible_findings:
                                        with st.container():
                                            st.markdown(f"📋 {finding}")
                                    
                                    # Show hidden findings with a toggle
                                    if hidden_findings:
                                        show_more = st.button(
                                            f"+ {len(hidden_findings)} more findings",
                                            key=f"findings_{i}"
                                        )
                                        if show_more:
                                            for finding in hidden_findings:
                                                st.markdown(f"📋 {finding}")
                            else:
                                st.info("No findings recorded")
                        
                        st.divider()
                        
                        # Recommendations section
                        st.markdown("##### Recommended Actions")
                        rec_cols = st.columns(2)
                        
                        def get_action_emoji(action):
                            if 'appointment' in action.lower():
                                return '📅'
                            elif 'medication' in action.lower():
                                return '💊'
                            elif 'weight' in action.lower():
                                return '⚖️'
                            elif 'exercise' in action.lower():
                                return '🏃'
                            elif 'diet' in action.lower():
                                return '🥗'
                            else:
                                return '📝'
                        
                        with rec_cols[0]:
                            st.markdown("**Clinical Recommendations:**")
                            for rec in patient.get('recommendations', []):
                                emoji = get_action_emoji(rec)
                                col1, col2 = st.columns([4, 1])
                                with col1:
                                    st.warning(f"{emoji} {rec}")
                                with col2:
                                    if st.button("Book Now →", key=f"clinical_{i}_{rec[:10]}"):
                                        st.success("✅ Booked!")
                        
                        with rec_cols[1]:
                            st.markdown("**Weight Management Recommendations:**")
                            for rec in weight_data.get('recommendations', []):
                                emoji = get_action_emoji(rec)
                                col1, col2 = st.columns([4, 1])
                                with col1:
                                    st.warning(f"{emoji} {rec}")
                                with col2:
                                    if st.button("Book Now →", key=f"weight_{i}_{rec[:10]}"):
                                        st.success("✅ Booked!")
                    
                    # Add spacing between cards
                    st.divider()
                
            except Exception as e:
                st.error(f"Error displaying patient: {str(e)}")
                print(f"Detailed error: {e}")
    
    with tab2:
        st.markdown("### All Patients")
        
        # Create DataFrame for AgGrid
        df_display = create_aggrid_table(data, priority_list)
        
        # Configure and display AgGrid
        gb = GridOptionsBuilder.from_dataframe(df_display)
        gb.configure_pagination()
        gb.configure_side_bar()
        
        # Configure column widths and formatting
        gb.configure_column("First Name", minWidth=120)
        gb.configure_column("Last Name", minWidth=120)
        gb.configure_column("Risk Score", 
                          minWidth=100,
                          maxWidth=120,
                          type=["numericColumn", "numberColumnFilter"],
                          valueFormatter="''+data['Risk Score']+'%'",
                          cellStyle={
                              "styleConditions": [
                                  {
                                      "condition": "params.value > 70",
                                      "style": {"backgroundColor": "#ff4444", "color": "white"}
                                  },
                                  {
                                      "condition": "params.value > 40",
                                      "style": {"backgroundColor": "#ffa726", "color": "black"}
                                  },
                                  {
                                      "condition": "params.value <= 40",
                                      "style": {"backgroundColor": "#66bb6a", "color": "black"}
                                  }
                              ]
                          })
        gb.configure_column("Risk Factors", minWidth=300)
        gb.configure_column("Recommendations", minWidth=300)
        gb.configure_column("State", maxWidth=100)
        gb.configure_column("Days Since Visit", 
                          minWidth=120,
                          maxWidth=150,
                          type=["numericColumn", "numberColumnFilter"])
        
        gb.configure_selection('single')
        gridOptions = gb.build()
        
        grid_response = AgGrid(
            df_display,
            gridOptions=gridOptions,
            data_return_mode='AS_INPUT',
            update_mode='MODEL_CHANGED',
            fit_columns_on_grid_load=True,
            theme='streamlit',
            height=400,
            allow_unsafe_jscode=True
        )

@st.cache_data
def get_cached_analysis():
    return analyze_patients_ml()

@st.cache_data
def get_cached_weight_analysis():
    """Cache the weight management analysis results"""
    from weight_management_analysis import analyze_weight_management
    return analyze_weight_management()

if __name__ == "__main__":
    main() 