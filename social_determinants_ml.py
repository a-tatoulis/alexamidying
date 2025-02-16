import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import os
from typing import List, Dict, Any

class SocialDeterminantsAnalyzer:
    def __init__(self):
        self.label_encoders = {}
        self.model = None
        
    def prepare_timeline_features(self, df):
        """
        Transform timeline data into features for each patient
        """
        try:
            # Validate input data
            required_columns = ['PATIENT', 'DATE', 'stress_level', 'employment_status', 
                              'housing_status', 'social_contact']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Convert date to datetime with error handling
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            if df['DATE'].isna().any():
                raise ValueError("Invalid date formats found in DATE column")
            
            # Create patient-level features
            patient_features = []
            
            for patient in df['PATIENT'].unique():
                patient_data = df[df['PATIENT'] == patient].sort_values('DATE')
                
                # Calculate timeline length
                timeline_length = (patient_data['DATE'].max() - patient_data['DATE'].min()).days
                
                # Encode categorical variables
                for col in ['stress_level', 'employment_status', 'housing_status', 'social_contact']:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        # Handle missing values before encoding
                        df[col] = df[col].fillna('unknown')
                        # Ensure 'unknown' is in the classes
                        unique_values = list(df[col].unique())
                        if 'unknown' not in unique_values:
                            unique_values.append('unknown')
                        self.label_encoders[col].fit(unique_values)
                        patient_data[f'{col}_encoded'] = self.label_encoders[col].transform(patient_data[col])
                    else:
                        # Handle missing values
                        patient_data[col] = patient_data[col].fillna('unknown')
                        # Transform using existing encoder
                        try:
                            patient_data[f'{col}_encoded'] = self.label_encoders[col].transform(patient_data[col])
                        except ValueError:
                            # If unknown values found, map them to 'unknown'
                            patient_data.loc[~patient_data[col].isin(self.label_encoders[col].classes_), col] = 'unknown'
                            patient_data[f'{col}_encoded'] = self.label_encoders[col].transform(patient_data[col])
                
                # Calculate features
                features = {
                    'patient_id': patient,
                    'timeline_length': timeline_length,
                    
                    # Stress level features
                    'stress_level_changes': len(patient_data['stress_level'].unique()),
                    'high_stress_ratio': (patient_data['stress_level'] == 'high').mean(),
                    
                    # Employment stability
                    'employment_changes': len(patient_data['employment_status'].unique()),
                    'employed_ratio': patient_data['employment_status'].isin(['full-time', 'part-time']).mean(),
                    
                    # Housing stability
                    'housing_changes': len(patient_data['housing_status'].unique()),
                    'stable_housing_ratio': (patient_data['housing_status'] == 'stable').mean(),
                    
                    # Social contact
                    'social_contact_changes': len(patient_data['social_contact'].unique()),
                    'good_social_contact_ratio': (patient_data['social_contact'] == 'regular').mean(),
                    
                    # Trend features
                    'stress_trend': patient_data['stress_level_encoded'].diff().mean(),
                    'social_contact_trend': patient_data['social_contact_encoded'].diff().mean(),
                    
                    # Volatility features
                    'stress_volatility': patient_data['stress_level_encoded'].std(),
                    'housing_volatility': patient_data['housing_status_encoded'].std(),
                    
                    # Recent state (last record)
                    'latest_stress': patient_data['stress_level'].iloc[-1],
                    'latest_employment': patient_data['employment_status'].iloc[-1],
                    'latest_housing': patient_data['housing_status'].iloc[-1],
                    'latest_social_contact': patient_data['social_contact'].iloc[-1]
                }
                
                patient_features.append(features)
            
            return pd.DataFrame(patient_features)
        except Exception as e:
            print(f"Error in prepare_timeline_features: {e}")
            return None
    
    def calculate_risk_score(self, patient_features):
        """
        Calculate risk score based on social determinants
        """
        risk_factors = {
            'stress': {
                'high_stress_ratio': 0.3,
                'stress_volatility': 0.2,
                'stress_trend': 0.1
            },
            'stability': {
                'employment_changes': 0.15,
                'housing_changes': 0.15,
                'employed_ratio': -0.2,
                'stable_housing_ratio': -0.2
            },
            'social': {
                'social_contact_changes': 0.1,
                'good_social_contact_ratio': -0.2
            }
        }
        
        risk_scores = []
        for _, patient in patient_features.iterrows():
            score = 0
            
            # Avoid division by zero for timeline_length
            timeline_length = max(patient['timeline_length'], 1)  # Use at least 1 day
            
            # Calculate stress risk
            score += patient['high_stress_ratio'] * risk_factors['stress']['high_stress_ratio']
            score += patient['stress_volatility'] * risk_factors['stress']['stress_volatility']
            score += patient['stress_trend'] * risk_factors['stress']['stress_trend']
            
            # Calculate stability risk with safe division
            score += (patient['employment_changes']/timeline_length) * risk_factors['stability']['employment_changes']
            score += (patient['housing_changes']/timeline_length) * risk_factors['stability']['housing_changes']
            score += patient['employed_ratio'] * risk_factors['stability']['employed_ratio']
            score += patient['stable_housing_ratio'] * risk_factors['stability']['stable_housing_ratio']
            
            # Calculate social risk
            score += (patient['social_contact_changes']/timeline_length) * risk_factors['social']['social_contact_changes']
            score += patient['good_social_contact_ratio'] * risk_factors['social']['good_social_contact_ratio']
            
            # Normalize score to 0-100 range
            score = max(min((score + 1) * 50, 100), 0)
            
            risk_scores.append({
                'patient_id': patient['patient_id'],
                'risk_score': score,
                'key_factors': self.identify_key_risk_factors(patient)
            })
        
        return risk_scores
    
    def identify_key_risk_factors(self, patient):
        """
        Identify key risk factors for a patient
        """
        risk_factors = []
        
        # Check stress factors
        if patient['high_stress_ratio'] > 0.5:
            risk_factors.append({
                'factor': 'High Stress Levels',
                'severity': 'high' if patient['high_stress_ratio'] > 0.7 else 'medium'
            })
            
        if patient['stress_volatility'] > 1:
            risk_factors.append({
                'factor': 'Unstable Stress Levels',
                'severity': 'medium'
            })
        
        # Check stability factors
        if patient['employment_changes'] > 2:
            risk_factors.append({
                'factor': 'Employment Instability',
                'severity': 'high'
            })
            
        if patient['housing_changes'] > 1:
            risk_factors.append({
                'factor': 'Housing Instability',
                'severity': 'high'
            })
        
        # Check social factors
        if patient['good_social_contact_ratio'] < 0.3:
            risk_factors.append({
                'factor': 'Low Social Contact',
                'severity': 'medium'
            })
        
        return risk_factors
    
    def generate_recommendations(self, risk_assessment):
        """
        Generate intervention recommendations based on risk assessment
        """
        recommendations = []
        
        for factor in risk_assessment['key_factors']:
            if factor['factor'] == 'High Stress Levels':
                recommendations.append({
                    'type': 'stress_management',
                    'priority': factor['severity'],
                    'description': 'Enroll in stress management program',
                    'frequency': 'weekly' if factor['severity'] == 'high' else 'bi-weekly'
                })
                
            elif factor['factor'] in ['Employment Instability', 'Housing Instability']:
                recommendations.append({
                    'type': 'stability_support',
                    'priority': factor['severity'],
                    'description': 'Connect with social worker for stability support',
                    'frequency': 'immediate consultation'
                })
                
            elif factor['factor'] == 'Low Social Contact':
                recommendations.append({
                    'type': 'social_support',
                    'priority': factor['severity'],
                    'description': 'Enroll in community support group',
                    'frequency': 'weekly'
                })
        
        return recommendations

# Example usage
def main():
    try:
        # Load timeline data with error handling
        if not os.path.exists('artifacts_data_detailedsocialdeterminantstimelinesamplepatients.csv'):
            raise FileNotFoundError("Timeline data file not found")
            
        data = pd.read_csv('artifacts_data_detailedsocialdeterminantstimelinesamplepatients.csv')
        
        # Validate data is not empty
        if data.empty:
            raise ValueError("Input data is empty")
            
        # Initialize analyzer
        analyzer = SocialDeterminantsAnalyzer()
        
        # Prepare features
        patient_features = analyzer.prepare_timeline_features(data)
        
        if patient_features is None or patient_features.empty:
            raise ValueError("No valid patient features could be extracted")
        
        # Calculate risk scores
        risk_assessments = analyzer.calculate_risk_score(patient_features)
        
        if not risk_assessments:
            raise ValueError("No risk assessments could be generated")
            
        # Generate recommendations for each patient
        for assessment in risk_assessments:
            print(f"\nPatient: {assessment['patient_id']}")
            print(f"Risk Score: {assessment['risk_score']:.1f}")
            print("\nKey Risk Factors:")
            for factor in assessment['key_factors']:
                print(f"- {factor['factor']} (Severity: {factor['severity']})")
            
            recommendations = analyzer.generate_recommendations(assessment)
            print("\nRecommended Interventions:")
            for rec in recommendations:
                print(f"- [{rec['priority']}] {rec['description']} ({rec['frequency']})")
                
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()