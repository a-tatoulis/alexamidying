import pandas as pd
import numpy as np
import duckdb
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class UnsupervisedAnalyzer:
    def __init__(self):
        # Create pipeline with imputer and scaler
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        self.kmeans = None
        self.pca = PCA(n_components=2)
        
    def prepare_patient_features(self, df):
        """Create feature set for each patient with proper missing value handling"""
        patient_features = []
        
        for patient in df['PATIENT'].unique():
            patient_data = df[df['PATIENT'] == patient].sort_values('observation_date')
            
            # Safe conversion functions
            def safe_mean(series):
                return np.nanmean(series) if len(series) > 0 else np.nan
                
            def safe_std(series):
                return np.nanstd(series) if len(series) > 0 else np.nan
                
            def safe_count_unique(series):
                return len(series.dropna().unique())
            
            # Convert stress levels to numeric safely
            stress_map = {
                'Not at all': 1,
                'A little bit': 2,
                'Somewhat': 3,
                'Quite a bit': 4,
                'Very much': 5
            }
            stress_numeric = patient_data['stress_level'].map(stress_map)
            
            # Convert social contact to numeric safely
            contact_map = {
                'Less than once a week': 1,
                '1 or 2 times a week': 2,
                '3 to 5 times a week': 3,
                '5 or more times a week': 4
            }
            contact_numeric = patient_data['social_contact_frequency'].map(contact_map)
            
            # Calculate dates safely
            try:
                dates = pd.to_datetime(patient_data['observation_date'])
                last_visit = (pd.Timestamp.now() - dates.max()).days
                visit_gaps = dates.diff().dt.days
                avg_gap = visit_gaps.mean()
            except:
                last_visit = np.nan
                avg_gap = np.nan
            
            features = {
                'patient': patient,
                # Visit patterns
                'total_visits': len(patient_data),
                'days_since_last_visit': last_visit,
                'avg_days_between_visits': avg_gap,
                
                # Stress patterns
                'mean_stress': safe_mean(stress_numeric),
                'stress_volatility': safe_std(stress_numeric),
                
                # Social contact patterns
                'mean_social_contact': safe_mean(contact_numeric),
                'social_contact_volatility': safe_std(contact_numeric),
                
                # Housing stability
                'housing_worry_ratio': (patient_data['housing_worry'] == 'Yes').mean(),
                
                # Employment
                'employment_changes': safe_count_unique(patient_data['employment_status']),
                
                # Recent patterns (last 3 observations if available)
                'recent_stress': safe_mean(stress_numeric.tail(3)),
                'recent_social_contact': safe_mean(contact_numeric.tail(3))
            }
            
            patient_features.append(features)
        
        return pd.DataFrame(patient_features)
    
    def identify_patterns(self, features_df):
        """Identify patterns with proper missing value handling"""
        # Prepare features for clustering
        features_for_clustering = features_df.drop(['patient'], axis=1)
        
        # Transform features using pipeline (imputation + scaling)
        transformed_features = self.pipeline.fit_transform(features_for_clustering)
        
        # Perform clustering
        self.kmeans = KMeans(n_clusters=min(4, len(features_df)), random_state=42)
        cluster_labels = self.kmeans.fit_predict(transformed_features)
        
        # PCA for visualization (after imputation)
        pca_coords = self.pca.fit_transform(transformed_features)
        
        # Add results to features
        features_df['cluster'] = cluster_labels
        features_df['pca_x'] = pca_coords[:, 0]
        features_df['pca_y'] = pca_coords[:, 1]
        
        return features_df
    
    def calculate_priority_score(self, row):
        """Calculate priority score with missing value handling"""
        scores = []
        weights = []
        
        # Stress score (30%)
        if pd.notnull(row['mean_stress']):
            stress_score = row['mean_stress'] / 5  # Normalize to 0-1
            if pd.notnull(row['stress_volatility']):
                stress_score = (stress_score + min(row['stress_volatility'], 1)) / 2
            scores.append(stress_score)
            weights.append(0.3)
            
        # Social contact score (25%)
        if pd.notnull(row['mean_social_contact']):
            social_score = 1 - (row['mean_social_contact'] / 4)  # Invert and normalize
            scores.append(social_score)
            weights.append(0.25)
            
        # Housing worry score (20%)
        if pd.notnull(row['housing_worry_ratio']):
            scores.append(row['housing_worry_ratio'])
            weights.append(0.2)
            
        # Visit gap score (15%)
        if pd.notnull(row['days_since_last_visit']):
            visit_gap_score = min(row['days_since_last_visit'] / 60, 1)  # Cap at 60 days
            scores.append(visit_gap_score)
            weights.append(0.15)
            
        # Employment changes score (10%)
        if pd.notnull(row['employment_changes']):
            emp_change_score = min(row['employment_changes'] / 2, 1)  # Cap at 2 changes
            scores.append(emp_change_score)
            weights.append(0.1)
        
        # Calculate weighted average if we have any scores
        if scores:
            # Normalize weights
            weights = [w/sum(weights) for w in weights]
            return sum(s * w for s, w in zip(scores, weights))
        else:
            return 0.0  # Return 0 instead of NaN for missing data
    
    def prioritize_patients(self, df_with_patterns):
        """Create prioritized list of patients with missing value handling"""
        # Calculate priority scores
        df_with_patterns['priority_score'] = df_with_patterns.apply(
            self.calculate_priority_score, axis=1
        )
        
        prioritized_patients = []
        
        for _, patient in df_with_patterns.sort_values('priority_score', ascending=False).iterrows():
            risk_factors = []
            recommendations = []
            
            # Identify risk factors and generate actionable recommendations
            if pd.notnull(patient['mean_stress']) and patient['mean_stress'] > 3:
                risk_factors.append({
                    'factor': 'High Stress Levels',
                    'severity': 'high' if patient['mean_stress'] > 4 else 'medium',
                    'value': patient['mean_stress']
                })
                recommendations.append('Initiate stress management protocol: Schedule 1:1 mindfulness training session')
                
            if pd.notnull(patient['mean_social_contact']) and patient['mean_social_contact'] < 2:
                risk_factors.append({
                    'factor': 'Low Social Contact',
                    'severity': 'high' if patient['mean_social_contact'] < 1.5 else 'medium',
                    'value': patient['mean_social_contact']
                })
                recommendations.append('Register for community wellness program: Focus on group activities and social engagement')
                
            if pd.notnull(patient['housing_worry_ratio']) and patient['housing_worry_ratio'] > 0.5:
                risk_factors.append({
                    'factor': 'Housing Concerns',
                    'severity': 'high',
                    'value': patient['housing_worry_ratio']
                })
                recommendations.append('Urgent social worker referral: Assess housing stability and provide resources')
                
            if pd.notnull(patient['employment_changes']) and patient['employment_changes'] > 1:
                risk_factors.append({
                    'factor': 'Employment Instability',
                    'severity': 'high' if patient['employment_changes'] > 2 else 'medium',
                    'value': patient['employment_changes']
                })
                recommendations.append('Schedule career counseling session: Address employment stability and skill development')
                
            if pd.notnull(patient['days_since_last_visit']) and patient['days_since_last_visit'] > 30:
                risk_factors.append({
                    'factor': 'Extended Gap in Care',
                    'severity': 'high' if patient['days_since_last_visit'] > 60 else 'medium',
                    'value': patient['days_since_last_visit']
                })
                recommendations.append('Priority follow-up required: Schedule comprehensive health assessment')
            
            prioritized_patients.append({
                'patient': patient['patient'],
                'priority_score': patient['priority_score'],
                'cluster': patient['cluster'],
                'days_since_visit': patient['days_since_last_visit'],
                'risk_factors': risk_factors,
                'recommendations': recommendations
            })
        
        return prioritized_patients

def analyze_patients_ml():
    """Main analysis function with DuckDB integration"""
    try:
        # Create database connection
        conn = duckdb.connect()
        
        # Register CSV files
        conn.execute("CREATE TABLE csv_data_observations AS SELECT * FROM read_csv_auto('csv/observations.csv')")
        conn.execute("CREATE TABLE csv_data_conditions AS SELECT * FROM read_csv_auto('csv/conditions.csv')")
        
        # Query to get social determinants data
        query = """
        WITH patient_timeline AS (
            SELECT 
                o.PATIENT,
                o.DATE as observation_date,
                MAX(CASE WHEN o.DESCRIPTION = 'Stress level' THEN o.VALUE END) as stress_level,
                MAX(CASE WHEN o.DESCRIPTION = 'Employment status - current' THEN o.VALUE END) as employment_status,
                MAX(CASE WHEN o.DESCRIPTION = 'Housing status' THEN o.VALUE END) as housing_status,
                MAX(CASE WHEN o.DESCRIPTION = 'Are you worried about losing your housing?' THEN o.VALUE END) as housing_worry,
                MAX(CASE WHEN LOWER(o.DESCRIPTION) LIKE '%see or talk to people%' THEN o.VALUE END) as social_contact_frequency
            FROM csv_data_observations o
            WHERE o.TYPE in ('text', 'numeric')
            AND (
                o.DESCRIPTION = 'Stress level'
                OR o.DESCRIPTION = 'Employment status - current'
                OR o.DESCRIPTION = 'Housing status'
                OR o.DESCRIPTION = 'Are you worried about losing your housing?'
                OR LOWER(o.DESCRIPTION) LIKE '%see or talk to people%'
            )
            GROUP BY o.PATIENT, o.DATE
            HAVING COUNT(DISTINCT o.DESCRIPTION) >= 3
        )
        SELECT * FROM patient_timeline
        ORDER BY PATIENT, observation_date;
        """
        
        # Execute query and create DataFrame
        df = conn.execute(query).df()
        conn.close()

        print("DataFrame columns:", df.columns.tolist())
        print("Sample data:", df.head())

        # Initialize analyzer
        analyzer = UnsupervisedAnalyzer()
        
        # Prepare features
        print("Preparing patient features...")
        patient_features = analyzer.prepare_patient_features(df)
        
        if patient_features.empty:
            print("No patient features were generated")
            return pd.DataFrame(), [], []
            
        # Identify patterns
        print("Identifying patterns...")
        patterns_df = analyzer.identify_patterns(patient_features)
        
        if patterns_df.empty:
            print("No patterns were identified")
            return pd.DataFrame(), [], []
            
        # Prioritize patients
        print("Prioritizing patients...")
        priority_list = analyzer.prioritize_patients(patterns_df)
        
        if not priority_list:
            print("No priority list was generated")
            return pd.DataFrame(), [], []
            
        print(f"Successfully processed {len(patterns_df)} patients")
        print(f"Generated {len(priority_list)} priority entries")
        
        return patterns_df, [], priority_list
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), [], []

if __name__ == "__main__":
    patterns_df, insights, priority_list = analyze_patients_ml()
    
    if not patterns_df.empty:
        print("\nAnalysis completed successfully!")
        print(f"Processed {len(patterns_df)} patients")
        print(f"Identified {len(patterns_df['cluster'].unique())} clusters")
        
        print("\nTop Priority Patients:")
        for patient in priority_list[:5]:
            print(f"\nPatient {patient['patient']}:")
            print(f"Priority Score: {patient['priority_score']:.2f}")
            print(f"Days Since Last Visit: {patient['days_since_visit']:.0f}")
            print("Risk Factors:")
            for factor in patient['risk_factors']:
                print(f"  - {factor['factor']} (Severity: {factor['severity']}, Value: {factor['value']})")
            print("Recommendations:")
            for recommendation in patient['recommendations']:
                print(f"  - {recommendation}")
    else:
        print("Analysis failed to complete")