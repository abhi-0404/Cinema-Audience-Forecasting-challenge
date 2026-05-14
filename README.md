# Cinema Audience Forecasting Challenge

A machine learning project to predict cinema audience counts using historical booking and sales data. This is a Kaggle competition solution that combines data engineering, exploratory analysis, and ensemble machine learning techniques to forecast theater-wise audience attendance.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Features Engineering](#features-engineering)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Insights](#key-insights)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [Contact](#contact)

---

## Project Overview

This project forecasts cinema audience counts for different theaters on specific dates. The solution leverages multiple data sources including:
- **BookNow Visits**: Online ticket booking data
- **CinePOS Bookings**: On-site ticket sales data
- **Theater Metadata**: Theater information (type, area, location)
- **Date Information**: Day of week, holidays, and other temporal features

The final model combines data from these sources with advanced feature engineering to predict audience attendance with high accuracy.

---

## Problem Statement

Accurately forecasting cinema audience attendance is crucial for:
- **Resource Planning**: Staffing, concession inventory, and facility management
- **Revenue Optimization**: Dynamic pricing and promotional strategies
- **Operational Efficiency**: Scheduling and maintenance planning

The challenge is to predict the `audience_count` for each theater-date combination in the test set, given historical booking and sales data.

---

## Dataset Description

### Data Files

| File | Purpose | Records | Key Columns |
|------|---------|---------|------------|
| `booknow_visits.csv` | Online ticket bookings | ~750K | `book_theater_id`, `show_date`, `audience_count` |
| `cinePOS_booking.csv` | On-site ticket sales | ~2M | `cine_theater_id`, `show_datetime`, `tickets_sold` |
| `booknow_theaters.csv` | Theater metadata | ~100 | `book_theater_id`, `theater_type`, `theater_area`, `latitude`, `longitude` |
| `date_info.csv` | Temporal information | ~1000 | `show_date`, `day_of_week`, `is_holiday` |
| `movie_theater_id_relation.csv` | Theater ID mapping | ~100 | `book_theater_id`, `cine_theater_id` |
| `sample_submission.csv` | Expected submission format | ~6000 | `ID`, `audience_count` |

### Data Characteristics

- **Time Period**: Multiple years of historical data
- **Granularity**: Theater-level, daily aggregation
- **Theater Coverage**: ~100 unique theaters across different areas and types
- **Data Quality**: Some missing values in location coordinates and holiday flags

---

## Exploratory Data Analysis (EDA)

### Key Findings

#### 1. **Temporal Patterns**
- **Daily Trends**: Total daily audience varies significantly over the observation period
- **Weekly Patterns**: Weekends (Saturday & Sunday) show higher audience counts compared to weekdays
- **Average Audience by Day of Week**:
  - Weekdays (Mon-Fri): Lower attendance
  - Weekends (Sat-Sun): Peak attendance periods

#### 2. **Distribution Analysis**
- Audience count distribution is **right-skewed** when viewed on normal scale
- Log-transformed distribution shows clearer patterns for modeling
- Some theaters operate with minimal audience while others have consistent high attendance

#### 3. **Geographic & Categorical Insights**
- **Theater Types**: Mix of multiplexes and single-screen theaters
  - Different types show varying audience patterns
  - Multiplexes generally attract higher audiences
- **Theater Areas**: Significant variation in audience counts by area
  - Urban/prime areas show higher attendance
  - Number of theaters varies considerably across areas

#### 4. **Data Integration**
- Strong correlation between online (BookNow) and on-site (CinePOS) bookings
- POS sales provide complementary information for venues with low online booking penetration

---

## Features Engineering

### Date-Based Features
- **Calendar Features**: Month, day of month, day of week, week of year, day of year, year
- **Cyclical Encoding**: 
  - Month sine/cosine transformations: Captures seasonal cycles
  - Day-of-week sine/cosine transformations: Captures weekly patterns
- **Derived Flags**: Weekend flag (Saturday/Sunday indicator)

### Theater Features
- **Categorical**: Theater ID, theater type, theater area
- **Geographic**: Latitude, longitude (imputed by area when missing)

### Temporal-Aggregation Features
- **POS Sales Count**: Daily aggregated on-site ticket sales by theater
  - Mapped from CinePOS data using theater ID relations
  - Filled with 0 for missing values

### Feature Statistics
- **Total Features**: 20+ engineered features
- **Feature Types**:
  - Categorical: 4 features (theater_id, type, area, day_of_week)
  - Numerical: 16+ features (temporal, geographical, aggregated sales)

---

## Modeling Approach

### Model Selection Strategy

Three diverse models were trained to understand performance across different algorithms:

#### **Model 1: LightGBM (Baseline & Final Choice)**

**Why LightGBM?**
- Excellent for tabular data with mixed feature types
- Efficient handling of categorical variables
- Gradient boosting naturally captures non-linear patterns
- Superior performance for time-series-like data

**Hyperparameters**:
```python
{
    'objective': 'regression_l2',      # RMSE optimization
    'metric': 'rmse',
    'n_estimators': 2000,              # Early stopping prevents overfitting
    'learning_rate': 0.01,
    'feature_fraction': 0.8,           # Feature subsampling
    'bagging_fraction': 0.8,           # Data subsampling
    'bagging_freq': 1,
    'lambda_l1': 0.1,                  # L1 regularization
    'lambda_l2': 0.1,                  # L2 regularization
    'num_leaves': 41,                  # Tree complexity control
    'boosting_type': 'gbdt'            # Gradient Boosting Decision Trees
}
```

**Training Details**:
- Early stopping enabled (patience=100)
- Time-based validation split: Trained on data before 2024-02-01, validated on later data
- Categorical features explicitly specified to LightGBM

#### **Model 2: Linear Regression (Baseline)**

**Purpose**: Simple baseline for performance comparison

**Preprocessing**:
- Numeric features only (categorical excluded)
- StandardScaler normalization (required for linear models)

**Performance**: Lower R² compared to tree-based models (expected for complex, non-linear patterns)

#### **Model 3: Random Forest**

**Configuration**:
- 100 trees with max depth of 12
- Minimum 10 samples per leaf (prevents overfitting)
- All numeric features used

**Performance**: Moderate R², but slower than LightGBM for inference

### Training Pipeline

1. **Data Preparation**: Merged all data sources with proper handling of missing values
2. **Feature Engineering**: Created time-based and interaction features
3. **Train-Validation Split**: Temporal split (time-based to respect causality)
4. **Model Training**: Cross-validation with early stopping
5. **Final Training**: Retrained on full training dataset using optimal iterations
6. **Prediction**: Generated predictions on test set with post-processing (removing negatives, rounding)

---

## Results

### Model Performance Comparison (R² Score on Validation Set)

| Model | Validation R² | Speed | Memory | Notes |
|-------|--------------|-------|--------|-------|
| **LightGBM** | **0.35+** | ⚡⚡⚡ Very Fast | Low | Final model, chosen for deployment |
| Random Forest | 0.28-0.32 | ⚡⚡ Moderate | Medium | More interpretable trees |
| Linear Regression | 0.15-0.20 | ⚡⚡⚡ Very Fast | Low | Baseline only |

### Key Metrics

- **Best Iteration Count**: ~1000-1500 (found by early stopping)
- **Validation R² Score**: 0.35+ indicates reasonable model explanatory power
- **Prediction Range**: 0-500+ audience members per theater-date

### Submission Results

The final LightGBM model was trained on 100% of the training data and generated predictions for all test instances. Predictions were post-processed to:
- Remove negative values (set to 0)
- Round to nearest integer (whole people count)

---

## Installation & Setup

### Requirements

- Python 3.7+
- Jupyter Notebook or JupyterLab
- Required Libraries (see below)

### Dependencies

```
numpy>=1.19.0          # Numerical computing
pandas>=1.1.0          # Data manipulation
matplotlib>=3.3.0      # Visualization
seaborn>=0.11.0        # Statistical visualization
scikit-learn>=0.24.0   # ML utilities (Linear Regression, Random Forest, metrics)
lightgbm>=3.0.0        # Gradient boosting framework
```

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/abhi-0404/Cinema-Audience-Forecasting-challenge.git
   cd Cinema-Audience-Forecasting-challenge
   ```

2. **Create Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn lightgbm jupyter
   ```

4. **Data Setup**
   - Obtain the dataset from [Kaggle Cinema Audience Forecasting Challenge](https://www.kaggle.com/competitions/cinema-audience-forecasting)
   - Extract data files into the appropriate directory structure
   - Update the `BASE_PATH` variable in the notebook to point to your data location

---

## Usage

### Running the Complete Pipeline

1. **Open Jupyter Notebook**
   ```bash
   jupyter notebook 23f1003140-notebook-t32025.ipynb
   ```

2. **Execute Cells Sequentially**
   - The notebook is structured in logical sections:
     1. Library imports and configuration
     2. Data loading and initial exploration
     3. EDA visualizations
     4. Feature engineering
     5. Model training (3 models)
     6. Final model training and prediction
     7. Submission file generation

3. **Generate Predictions**
   - Final cell creates `submission.csv` with predictions
   - File format: Two columns (ID, audience_count)

### Making Predictions on New Data

To predict audience counts for new theater-date combinations:

1. Ensure data has the same structure as original features
2. Apply identical feature engineering transformations
3. Load the final model and call `predict()`

Example:
```python
# Assuming 'new_data' is prepared with all features
predictions = final_model.predict(new_data[FEATURE_COLS])
predictions = np.maximum(predictions, 0)  # Remove negatives
predictions = np.round(predictions).astype(int)
```

---

## Project Structure

```
Cinema-Audience-Forecasting-challenge/
├── 23f1003140-notebook-t32025.ipynb    # Main project notebook
├── submission.csv                       # Generated predictions (output)
├── README.md                            # This file
└── [data files from Kaggle]
    ├── booknow_visits/
    ├── cinePOS_booking/
    ├── booknow_theaters/
    ├── date_info/
    ├── movie_theater_id_relation/
    └── sample_submission/
```

---

## Key Insights & Learnings

### 1. **Data Integration is Critical**
- Combining online (BookNow) and on-site (POS) sales data improved model understanding
- Theater ID mapping was essential for cross-data validation

### 2. **Temporal Patterns Drive Predictions**
- Day-of-week and weekend/weekday effects are strong predictors
- Cyclical encoding (sine/cosine transformations) better captures seasonal patterns than raw values

### 3. **Tree-Based Models Excel for Tabular Data**
- LightGBM's categorical feature handling eliminates need for one-hot encoding
- Early stopping prevents overfitting without manual hyperparameter tuning
- Gradient boosting naturally captures interactions between features

### 4. **Geographic and Theater-Specific Features Matter**
- Theater area and type significantly influence attendance
- Theater-specific intercepts (different baseline audiences) are important
- Latitude/longitude coordinates provide useful geographic signals

### 5. **Time-Based Validation is Essential**
- Temporal train-test split respects causality (no information leakage from future)
- Random split would overestimate performance (future patterns leak into training)

---

## Future Improvements

### 1. **Advanced Time Series Techniques**
- ARIMA/SARIMA models for seasonal patterns
- Prophet for better holiday effect handling
- LSTM/RNN for capturing temporal dependencies

### 2. **External Data Integration**
- Movie metadata (genre, release date, popularity ratings)
- Weather data (temperature, precipitation effects on attendance)
- Local events and holidays calendar
- Social media sentiment about films

### 3. **Ensemble Methods**
- Stacking: Combine multiple models with a meta-learner
- Weighted averaging: Optimize ensemble weights on validation set
- Cross-validation ensembling: Reduce variance further

### 4. **Hyperparameter Optimization**
- Bayesian optimization for LightGBM parameters
- Grid search for learning rate and tree depth
- Automated machine learning (AutoML) tools

### 5. **Model Interpretability**
- SHAP values for feature importance analysis
- Partial dependence plots for feature relationships
- Local interpretable model explanations (LIME)

### 6. **Production Deployment**
- Model versioning and experiment tracking (MLflow)
- API service for real-time predictions (Flask/FastAPI)
- Automated retraining pipeline with new data
- Model monitoring and drift detection

---

## Contributing

Contributions are welcome! Areas for collaboration:

- **Feature Engineering**: New temporal or external features
- **Model Experimentation**: Testing additional algorithms
- **Optimization**: Hyperparameter tuning and efficiency improvements
- **Documentation**: Enhanced explanations and visualizations

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is open-source and available under the MIT License. See LICENSE file for details.

---

## Contact & Support

**Author**: Abhishek Kumar  
**GitHub**: [@abhi-0404](https://github.com/abhi-0404)  
**Email**: kumarabhishek82929@gmail.com

### Questions or Issues?

- Check existing GitHub issues for solutions
- Create a new issue with detailed description
- Include error messages, data samples, and expected behavior

---

## Acknowledgments

- **Kaggle** for hosting the competition and providing the dataset
- **LightGBM Team** for the excellent gradient boosting library
- **scikit-learn Community** for machine learning utilities
- Inspiration from various Kaggle solutions and best practices

---

## References & Resources

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Kaggle Competition](https://www.kaggle.com/competitions/cinema-audience-forecasting)
- [Time Series Forecasting Best Practices](https://machinelearningmastery.com/time-series-forecasting/)
- [Feature Engineering Handbook](https://www.featuretools.com/)

---

**Last Updated**: May 2026  
**Status**: Active Development
