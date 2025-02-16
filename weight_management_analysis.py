import pandas as pd
import numpy as np
import duckdb
from datetime import datetime, timedelta

class WeightManagementAnalyzer:
    def __init__(self):
        self.conn = duckdb.connect()
        self.glp1_codes = [
            '897122',  # Liraglutide
            '897123',  # Semaglutide
            '897124',  # Tirzepatide
        ]
        
    def load_data(self):
        """Load and prepare data for analysis"""
        # Register all needed CSV files
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS csv_data_observations AS 
            SELECT * FROM read_csv_auto('csv/observations.csv')
            WHERE type in ('text', 'numeric')
            AND (
                description = 'Body Weight'
                OR description = 'Body Mass Index'
                OR description LIKE '%Goal%'
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS csv_data_medications AS 
            SELECT * FROM read_csv_auto('csv/medications.csv')
            WHERE code IN ('897122', '897123', '897124')  -- GLP-1 medication codes
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS csv_data_encounters AS 
            SELECT * FROM read_csv_auto('csv/encounters.csv')
            WHERE encounterclass IN ('ambulatory', 'wellness')
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS csv_data_conditions AS 
            SELECT * FROM read_csv_auto('csv/conditions.csv')
            WHERE stop IS NULL
        """)
        
        # Get obese patients with their diagnosis dates
        obese_patients = self.conn.execute("""
            WITH obese_patients AS (
                SELECT DISTINCT PATIENT, MIN(start) as diagnosis_date 
                FROM csv_data_conditions 
                WHERE LOWER(description) LIKE '%obese%' 
                   OR LOWER(description) LIKE '%bmi 30%' 
                   OR LOWER(description) LIKE '%bmi 40%'
                GROUP BY PATIENT
            )
            SELECT * FROM obese_patients
        """).df()
        
        # Get weight measurements for obese patients
        weight_data = self.conn.execute("""
            SELECT 
                o.PATIENT,
                o.date as measurement_date,
                CAST(o.value AS FLOAT) as weight_value,
                o.description
            FROM csv_data_observations o
            INNER JOIN obese_patients op ON o.PATIENT = op.PATIENT
            WHERE o.description = 'Body Weight'
            ORDER BY o.PATIENT, o.date
        """).df()
        
        # Get GLP-1 medication data
        medication_data = self.conn.execute("""
            SELECT 
                m.PATIENT,
                m.START as prescription_date,
                m.STOP as end_date,
                m.DISPENSES as dispenses,
                m.TOTALCOST as total_cost,
                m.CODE as code,
                m.DESCRIPTION as description
            FROM csv_data_medications m
            INNER JOIN obese_patients op ON m.PATIENT = op.PATIENT
            ORDER BY m.PATIENT, m.START
        """).df()
        
        # Get encounter data
        encounter_data = self.conn.execute("""
            SELECT 
                e.PATIENT,
                e.START as encounter_date,
                e.ENCOUNTERCLASS as encounter_class,
                e.DESCRIPTION as description
            FROM csv_data_encounters e
            INNER JOIN obese_patients op ON e.PATIENT = op.PATIENT
            WHERE e.ENCOUNTERCLASS IN ('ambulatory', 'wellness')
            ORDER BY e.PATIENT, e.START
        """).df()
        
        return obese_patients, weight_data, medication_data, encounter_data
    
    def calculate_weight_metrics(self, weight_data):
        """Calculate weight change metrics for each patient"""
        weight_metrics = []
        
        for patient in weight_data['PATIENT'].unique():
            patient_weights = weight_data[weight_data['PATIENT'] == patient].sort_values('measurement_date')
            
            if len(patient_weights) >= 2:
                initial_weight = patient_weights['weight_value'].iloc[0]
                current_weight = patient_weights['weight_value'].iloc[-1]
                max_weight = patient_weights['weight_value'].max()
                
                # Calculate total and percentage weight change
                total_change = current_weight - initial_weight
                pct_change = (total_change / initial_weight) * 100
                
                # Calculate rate of weight change (lbs per month)
                days_elapsed = (patient_weights['measurement_date'].iloc[-1] - 
                              patient_weights['measurement_date'].iloc[0]).days
                if days_elapsed > 0:
                    monthly_rate = (total_change / days_elapsed) * 30
                else:
                    monthly_rate = 0
                
                weight_metrics.append({
                    'PATIENT': patient,
                    'initial_weight': initial_weight,
                    'current_weight': current_weight,
                    'total_change': total_change,
                    'pct_change': pct_change,
                    'monthly_rate': monthly_rate,
                    'measurement_count': len(patient_weights),
                    'days_tracked': days_elapsed
                })
        
        return pd.DataFrame(weight_metrics)
    
    def calculate_medication_adherence(self, medication_data):
        """Calculate medication adherence metrics"""
        adherence_metrics = []
        
        for patient in medication_data['PATIENT'].unique():
            patient_meds = medication_data[medication_data['PATIENT'] == patient].sort_values('prescription_date')
            
            # Calculate total prescribed days
            total_days = 0
            total_cost = 0
            total_dispenses = 0
            active_prescription = False
            
            for _, med in patient_meds.iterrows():
                if pd.isnull(med['end_date']):
                    # If no end date, assume current
                    days = (datetime.now() - pd.to_datetime(med['prescription_date'])).days
                    active_prescription = True
                else:
                    days = (pd.to_datetime(med['end_date']) - pd.to_datetime(med['prescription_date'])).days
                
                total_days += max(0, days)
                total_cost += med['total_cost'] if pd.notnull(med['total_cost']) else 0
                total_dispenses += med['dispenses'] if pd.notnull(med['dispenses']) else 0
            
            adherence_metrics.append({
                'PATIENT': patient,
                'total_prescription_days': total_days,
                'total_dispenses': total_dispenses,
                'total_cost': total_cost,
                'active_prescription': active_prescription,
                'prescription_count': len(patient_meds)
            })
        
        return pd.DataFrame(adherence_metrics)
    
    def calculate_appointment_metrics(self, encounter_data):
        """Calculate appointment attendance metrics"""
        attendance_metrics = []
        
        for patient in encounter_data['PATIENT'].unique():
            patient_encounters = encounter_data[encounter_data['PATIENT'] == patient].sort_values('encounter_date')
            
            # Calculate days between visits
            visit_gaps = patient_encounters['encounter_date'].diff().dt.days
            
            # Calculate metrics
            metrics = {
                'PATIENT': patient,
                'total_visits': len(patient_encounters),
                'avg_days_between_visits': visit_gaps.mean(),
                'max_gap': visit_gaps.max(),
                'wellness_visits': len(patient_encounters[patient_encounters['encounter_class'] == 'wellness']),
                'days_since_last_visit': (datetime.now() - pd.to_datetime(patient_encounters['encounter_date'].max())).days
            }
            
            attendance_metrics.append(metrics)
        
        return pd.DataFrame(attendance_metrics)
    
    def calculate_progress_score(self, weight_metrics, adherence_metrics, attendance_metrics):
        """Calculate overall progress score incorporating positive factors"""
        progress_scores = []
        
        # Merge all metrics
        combined_metrics = weight_metrics.merge(
            adherence_metrics, on='PATIENT', how='outer'
        ).merge(
            attendance_metrics, on='PATIENT', how='outer'
        )
        
        for _, patient in combined_metrics.iterrows():
            positive_factors = []
            recommendations = []
            
            # Weight loss progress (40% of score)
            weight_score = 0
            if pd.notnull(patient['pct_change']):
                if patient['pct_change'] <= -5:  # Lost 5% or more
                    weight_score = 1.0
                    positive_factors.append({
                        'factor': 'Significant Weight Loss',
                        'impact': 'high',
                        'value': abs(patient['pct_change'])
                    })
                elif patient['pct_change'] <= -2:  # Lost 2-5%
                    weight_score = 0.7
                    positive_factors.append({
                        'factor': 'Moderate Weight Loss',
                        'impact': 'medium',
                        'value': abs(patient['pct_change'])
                    })
                elif patient['pct_change'] < 0:  # Any weight loss
                    weight_score = 0.4
                    positive_factors.append({
                        'factor': 'Initial Weight Loss',
                        'impact': 'low',
                        'value': abs(patient['pct_change'])
                    })
            
            # Medication adherence (30% of score)
            med_score = 0
            if pd.notnull(patient['total_prescription_days']):
                if patient['active_prescription'] and patient['total_prescription_days'] > 90:
                    med_score = 1.0
                    positive_factors.append({
                        'factor': 'Consistent Medication Adherence',
                        'impact': 'high',
                        'value': patient['total_prescription_days']
                    })
                elif patient['active_prescription']:
                    med_score = 0.7
                    positive_factors.append({
                        'factor': 'Active Medication Adherence',
                        'impact': 'medium',
                        'value': patient['total_prescription_days']
                    })
            
            # Appointment attendance (30% of score)
            visit_score = 0
            if pd.notnull(patient['total_visits']):
                if patient['days_since_last_visit'] <= 30 and patient['wellness_visits'] >= 2:
                    visit_score = 1.0
                    positive_factors.append({
                        'factor': 'Excellent Visit Adherence',
                        'impact': 'high',
                        'value': patient['total_visits']
                    })
                elif patient['days_since_last_visit'] <= 60:
                    visit_score = 0.7
                    positive_factors.append({
                        'factor': 'Good Visit Adherence',
                        'impact': 'medium',
                        'value': patient['total_visits']
                    })
                elif patient['days_since_last_visit'] <= 90:
                    visit_score = 0.4
                    positive_factors.append({
                        'factor': 'Regular Visits',
                        'impact': 'low',
                        'value': patient['total_visits']
                    })
            
            # Calculate weighted score
            total_score = (
                weight_score * 0.4 +
                med_score * 0.3 +
                visit_score * 0.3
            )
            
            # Generate recommendations
            if total_score < 0.4:
                recommendations.append("Schedule comprehensive weight management consultation")
                recommendations.append("Review medication options with healthcare provider")
            elif total_score < 0.7:
                if weight_score < 0.4:
                    recommendations.append("Consider adjusting weight loss strategy")
                if med_score < 0.7:
                    recommendations.append("Discuss medication adherence with healthcare provider")
                if visit_score < 0.7:
                    recommendations.append("Schedule follow-up appointment")
            else:
                recommendations.append("Continue current successful management plan")
                recommendations.append("Consider setting new health goals")
            
            progress_scores.append({
                'PATIENT': patient['PATIENT'],
                'progress_score': total_score,
                'positive_factors': positive_factors,
                'recommendations': recommendations,
                'weight_metrics': {
                    'total_change': patient['total_change'],
                    'pct_change': patient['pct_change'],
                    'monthly_rate': patient['monthly_rate']
                },
                'medication_metrics': {
                    'days_on_medication': patient['total_prescription_days'],
                    'active_prescription': patient['active_prescription']
                },
                'visit_metrics': {
                    'total_visits': patient['total_visits'],
                    'days_since_last': patient['days_since_last_visit']
                }
            })
        
        return progress_scores

def analyze_weight_management():
    """Main analysis function"""
    analyzer = WeightManagementAnalyzer()
    
    # Load data
    obese_patients, weight_data, medication_data, encounter_data = analyzer.load_data()
    
    # Calculate metrics
    weight_metrics = analyzer.calculate_weight_metrics(weight_data)
    adherence_metrics = analyzer.calculate_medication_adherence(medication_data)
    attendance_metrics = analyzer.calculate_appointment_metrics(encounter_data)
    
    # Calculate progress scores
    progress_scores = analyzer.calculate_progress_score(
        weight_metrics, adherence_metrics, attendance_metrics
    )
    
    return weight_metrics, adherence_metrics, attendance_metrics, progress_scores

if __name__ == "__main__":
    weight_metrics, adherence_metrics, attendance_metrics, progress_scores = analyze_weight_management()
    
    print(f"\nAnalyzed {len(progress_scores)} obese patients")
    
    # Print summary statistics
    high_progress = sum(1 for p in progress_scores if p['progress_score'] > 0.7)
    med_progress = sum(1 for p in progress_scores if 0.4 <= p['progress_score'] <= 0.7)
    low_progress = sum(1 for p in progress_scores if p['progress_score'] < 0.4)
    
    print("\nProgress Distribution:")
    print(f"High Progress (>70%): {high_progress} patients")
    print(f"Medium Progress (40-70%): {med_progress} patients")
    print(f"Low Progress (<40%): {low_progress} patients")
    
    # Print detailed results for top performers
    print("\nTop Performing Patients:")
    for patient in sorted(progress_scores, key=lambda x: x['progress_score'], reverse=True)[:5]:
        print(f"\nPatient ID: {patient['PATIENT']}")
        print(f"Progress Score: {patient['progress_score']*100:.1f}%")
        print("Positive Factors:")
        for factor in patient['positive_factors']:
            print(f"- {factor['factor']} (Impact: {factor['impact']}, Value: {factor['value']:.1f})")
        print("Recommendations:")
        for rec in patient['recommendations']:
            print(f"- {rec}") 