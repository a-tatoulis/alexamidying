# Patient Risk Dashboard

A modern, interactive dashboard for monitoring patient risk factors and interventions. Built with Streamlit and machine learning.

## Features

- Real-time risk assessment and scoring
- Interactive patient profiles with detailed metrics
- Automated intervention recommendations
- Data visualization with Plotly and Altair
- Responsive design with dark theme
- Efficient data caching for performance

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd patient-risk-dashboard
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit server:
```bash
streamlit run dashboard.py
```

2. Open your browser and navigate to `http://localhost:8501`

## Data Format

The dashboard expects patient data in a specific format. Ensure your data includes:
- Patient demographics (ID, name, age, etc.)
- Medical conditions
- Visit history
- Risk factors
- Intervention history

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 