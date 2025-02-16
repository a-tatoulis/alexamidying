import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime

class PatternDetector:
    def __init__(self):
        self.stress_map = {
            'Not at all': 1,
            'A little bit': 2,
            'Somewhat': 3,
            'Quite a bit': 4,
            'Very much': 5,
            'I choose not to answer this question': None
        }
        
        self.contact_map = {
            'Less than once a week': 1,
            '1 or 2 times a week': 2,
            '3 to 5 times a week': 3,
            '5 or more times a week': 4
        }
        
        self.employment_risk_map = {
            'Full-time work': 1,
            'Part-time or temporary work': 2,
            'Otherwise unemployed but not seeking work (ex: student  retired  disabled  unpaid primary care giver)': 3,
            'Unemployed': 4
        }

    def prepare_data(self, df):
        """Prepare the dataset for analysis"""
        print("Available columns:", df.columns.tolist())
        
        try:
            # Convert dates
            df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'], errors='coerce')
            df['observation_date'] = pd.to_datetime(df['observation_date'], errors='coerce')
            
            # Calculate days since diagnosis
            df['days_since_diagnosis'] = (df['observation_date'] - df['diagnosis_date']).dt.days

            # Map categorical variables to numeric
            df['stress_level_numeric'] = df['stress_level'].map(self.stress_map)
            
            # Handle social contact frequency
            df['social_contact_numeric'] = df['social_contact_frequency'].map(self.contact_map)
            
            # Handle employment status
            df['employment_risk'] = df['employment_status'].map(self.employment_risk_map)
            
            # Handle housing worry
            df['housing_worry_numeric'] = df['housing_worry'].map({'No': 0, 'Yes': 1})

            return df
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            return df

    def extract_patient_features(self, df):
        """Extract features for each patient"""
        patient_features = []
        
        for patient in df['patient'].unique():
            patient_data = df[df['patient'] == patient].sort_values('days_since_diagnosis')
            
            # Calculate stress features
            stress_data = patient_data['stress_level_numeric'].dropna()
            recent_stress = stress_data.tail(3).mean() if len(stress_data) >= 3 else stress_data.mean()
            
            # Calculate social contact features
            contact_data = patient_data['social_contact_numeric'].dropna()
            recent_contact = contact_data.tail(3).mean() if len(contact_data) >= 3 else contact_data.mean()
            
            features = {
                'patient': patient,
                # Stress patterns
                'mean_stress': stress_data.mean(),
                'recent_stress': recent_stress,
                'stress_volatility': stress_data.std(),
                'stress_trend': stress_data.diff().mean(),
                
                # Social contact patterns
                'mean_social_contact': contact_data.mean(),
                'recent_social_contact': recent_contact,
                'social_isolation_risk': (contact_data <= 2).mean(),
                
                # Employment patterns
                'employment_changes': len(patient_data['employment_status'].unique()),
                'recent_employment_risk': patient_data['employment_risk'].tail(3).mean(),
                
                # Housing stability
                'housing_worry_ratio': patient_data['housing_worry_numeric'].mean(),
                
                # Timeline characteristics
                'observation_count': len(patient_data),
                'observation_span': patient_data['days_since_diagnosis'].max()
            }
            
            patient_features.append(features)
        
        return pd.DataFrame(patient_features)

    def identify_patterns(self, patient_features):
        """Identify common patterns and risk groups"""
        patterns = []
        
        for _, patient in patient_features.iterrows():
            patient_patterns = {
                'patient': patient['patient'],
                'risk_factors': [],
                'risk_level': 'low',
                'trends': []
            }
            
            # Check stress patterns
            if patient['recent_stress'] > 3.5:
                patient_patterns['risk_factors'].append({
                    'factor': 'High Stress',
                    'severity': 'high',
                    'value': patient['recent_stress']
                })
            elif patient['stress_volatility'] > 1:
                patient_patterns['risk_factors'].append({
                    'factor': 'Unstable Stress',
                    'severity': 'medium',
                    'value': patient['stress_volatility']
                })
            
            # Check social isolation risk
            if patient['social_isolation_risk'] > 0.3:
                patient_patterns['risk_factors'].append({
                    'factor': 'Social Isolation Risk',
                    'severity': 'high' if patient['social_isolation_risk'] > 0.5 else 'medium',
                    'value': patient['social_isolation_risk']
                })
            
            # Check employment stability
            if patient['employment_changes'] > 1:
                patient_patterns['risk_factors'].append({
                    'factor': 'Employment Instability',
                    'severity': 'high' if patient['employment_changes'] > 2 else 'medium',
                    'value': patient['employment_changes']
                })
            
            # Check housing concerns
            if patient['housing_worry_ratio'] > 0:
                patient_patterns['risk_factors'].append({
                    'factor': 'Housing Concerns',
                    'severity': 'high' if patient['housing_worry_ratio'] > 0.5 else 'medium',
                    'value': patient['housing_worry_ratio']
                })
            
            # Analyze trends
            if patient['stress_trend'] > 0.2:
                patient_patterns['trends'].append('Increasing stress levels')
            elif patient['stress_trend'] < -0.2:
                patient_patterns['trends'].append('Decreasing stress levels')
            
            # Calculate overall risk level
            high_severity_count = sum(1 for factor in patient_patterns['risk_factors'] 
                                    if factor['severity'] == 'high')
            medium_severity_count = sum(1 for factor in patient_patterns['risk_factors'] 
                                      if factor['severity'] == 'medium')
            
            if high_severity_count >= 2 or (high_severity_count >= 1 and medium_severity_count >= 2):
                patient_patterns['risk_level'] = 'high'
            elif high_severity_count >= 1 or medium_severity_count >= 2:
                patient_patterns['risk_level'] = 'medium'
            
            patterns.append(patient_patterns)
        
        return patterns

    def generate_recommendations(self, patterns):
        """Generate recommendations based on identified patterns"""
        recommendations = []
        
        for patient in patterns:
            patient_recs = {
                'patient': patient['patient'],
                'risk_level': patient['risk_level'],
                'recommendations': []
            }
            
            # Generate specific recommendations based on risk factors
            for factor in patient['risk_factors']:
                if factor['factor'] == 'High Stress':
                    patient_recs['recommendations'].append({
                        'type': 'stress_management',
                        'priority': factor['severity'],
                        'description': 'Enroll in stress management program',
                        'frequency': 'weekly' if factor['severity'] == 'high' else 'bi-weekly'
                    })
                
                elif factor['factor'] == 'Social Isolation Risk':
                    patient_recs['recommendations'].append({
                        'type': 'social_support',
                        'priority': factor['severity'],
                        'description': 'Connect with community support group',
                        'frequency': 'weekly' if factor['severity'] == 'high' else 'bi-weekly'
                    })
                
                elif factor['factor'] in ['Employment Instability', 'Housing Concerns']:
                    patient_recs['recommendations'].append({
                        'type': 'stability_support',
                        'priority': factor['severity'],
                        'description': f"Connect with social worker for {factor['factor'].lower()} support",
                        'frequency': 'immediate consultation'
                    })
            
            recommendations.append(patient_recs)
        
        return recommendations

def analyze_patients(csv_path):
    """Main analysis function"""
    # Read data
    df = pd.read_csv(csv_path)
    
    # Initialize detector
    detector = PatternDetector()
    
    # Process data
    processed_df = detector.prepare_data(df)
    
    # Extract features
    patient_features = detector.extract_patient_features(processed_df)
    
    # Identify patterns
    patterns = detector.identify_patterns(patient_features)
    
    # Generate recommendations
    recommendations = detector.generate_recommendations(patterns)
    
    # Print summary
    print("\nAnalysis Summary:")
    risk_levels = {'high': 0, 'medium': 0, 'low': 0}
    for patient in patterns:
        risk_levels[patient['risk_level']] += 1
    
    print(f"\nRisk Level Distribution:")
    for level, count in risk_levels.items():
        print(f"{level.capitalize()}: {count} patients")
    
    print("\nDetailed Patient Patterns:")
    for patient in patterns:
        if patient['risk_factors']:  # Only show patients with identified risk factors
            print(f"\nPatient {patient['patient']}:")
            print(f"Risk Level: {patient['risk_level'].upper()}")
            print("Risk Factors:")
            for factor in patient['risk_factors']:
                print(f"- {factor['factor']}: {factor['severity'].upper()} (value: {factor['value']:.2f})")
            if patient['trends']:
                print("Trends:", ", ".join(patient['trends']))
    
    return patterns, recommendations

if __name__ == "__main__":
    patterns, recommendations = analyze_patients('obese_patients_social_determinants.csv')