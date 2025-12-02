Hotel Booking Cancellation Prediction — Machine Learning & Interpretability
_______________________________________________________________________________________________________________________________________
🧾 1. Project Overview

This project develops an interpretable machine learning model to predict hotel booking cancellations using historical booking data.

🎯 The model helps hotels:<br>

1] Reduce revenue loss<br>
2] Optimize overbooking strategy<br>
3] Improve resource allocation<br>
4] Understand cancellation behavior<br>

Repository Includes:<br>

✔ Exploratory Data Analysis (EDA)<br>
✔ Feature Engineering<br>
✔ ML Modeling (Logistic Regression, Random Forest, XGBoost)<br>
✔ SHAP Explainability<br>
✔ Prediction Pipeline<br>
📂 2. Project Structure
📁 Hotel Booking Cancellation Prediction
├── index.ipynb  ---------              # Main notebook (EDA + ML)<br>
├── Dataset.csv  ---------              # Dataset<br>
├── eda_outputs/ ---------              # Saved plots & analysis files<br>
├── models/      ---------              # Saved trained models<br>
├── README.md    ---------              # Documentation<br>
└── utils/       ---------              # Helper scripts (optional)<br>
__________________________________________________________________________________________________________________________________________
📊 3. Dataset Description

The dataset contains 10,000 hotel booking records with 28 features.

Key Columns:
Feature	Description
booking_date-------	Date when the booking was made<br>
arrival_date-------	Planned check-in date<br>
lead_time	Days----- between booking and arrival<br>
market_segment-----	Booking source<br>
booking_channel----	Distribution channel<br>
cancelled	---------Target variable (1 = cancelled)<br>
stay_length--------	Total nights stayed<br>
revenue------------	Revenue generated<br>

Target Variable:
cancelled = 1 → booking canceled<br>
cancelled = 0 → booking kept<br>

🔍 4. Exploratory Data Analysis (EDA)

The EDA notebook includes:

✔ Data Quality Checks
Missing values
Data type validation
Outlier identification
Duplicate detection

✔ Visualizations
Cancellation ratio
Monthly cancellation trends
Lead time distribution
Stay length distribution
Cancellation rates by:Market segment, Booking channel, Customer type.<br>

✔ Date Logic Validation
Ensures chronological correctness:
booking_date ≤ cancellation_date ≤ arrival_date
Violations are exported to: eda_outputs/date_logic_violations.csv<br>

