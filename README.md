# Real Estate Price Prediction

A comprehensive machine learning project for predicting house prices using the Ames Housing dataset. This project implements end-to-end data science workflows including data extraction, preprocessing, exploratory data analysis (EDA), feature engineering, and predictive modeling.

## Project Overview

This project is part of the Advanced Apex Project 1 at BITS Pilani Digital (First Trimester 2025-26), undertaken by Team **The Outliers**. The goal is to build accurate regression models that predict residential property prices based on 80+ features including physical attributes, location, quality ratings, and temporal factors.

### Key Objectives

- Perform comprehensive exploratory data analysis on the Ames Housing dataset
- Handle missing values, outliers, and data quality issues systematically
- Engineer meaningful features that improve predictive performance
- Build and compare multiple regression models
- Achieve accurate price predictions with interpretable results

## Dataset

**Source**: Ames Housing Dataset
**Records**: 2,930 residential property sales
**Original Features**: 82
**Target Variable**: `SalePrice` (house sale price in USD)

The dataset includes:
- **Physical Features**: Lot area, living area, number of rooms, garage capacity, basement size
- **Quality Metrics**: Overall quality, kitchen quality, exterior quality, basement quality
- **Categorical Features**: Neighborhood, building type, zoning, roof style, heating type
- **Temporal Features**: Year built, year remodeled, month/year sold

### Data Files

```
data/
├── AmesHousing.csv              # Original dataset (2930 × 82)
├── AmesHousing_cleaned.csv      # After preprocessing (2930 × 76)
└── AmesHousing_engineered.csv   # Ready for modeling (2930 × 270)
```

## Project Structure

```
real-estate-price-prediction/
├── data/                                  # Dataset files
│   ├── AmesHousing.csv
│   ├── AmesHousing_cleaned.csv
│   └── AmesHousing_engineered.csv
├── docs/                                  # Documentation
│   └── schema_summary.csv                 # Feature metadata
├── notebooks/                             # Phase-wise Jupyter notebooks
│   ├── 01_data_extraction.ipynb          # Data loading & validation
│   ├── 02_preprocessing_eda.ipynb        # Cleaning & EDA
│   ├── 03_feature_engineering.ipynb      # Feature creation
│   ├── 04_modeling.ipynb                 # Model training
│   └── 05_visualization_storytelling.ipynb # Results visualization
├── real_estate_price_prediction.ipynb     # Comprehensive master notebook
├── main.py                                # Main Python script
├── requirements.txt                       # Python dependencies
├── pyproject.toml                         # Project configuration
└── README.md                              # This file
```

## Technology Stack

### Languages & Libraries

- **Python 3.12+**: Core programming language
- **pandas 2.3.3**: Data manipulation and analysis
- **numpy 2.3.4**: Numerical computing
- **scikit-learn**: Machine learning models and preprocessing
- **matplotlib & seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development environment

### Key Tools

- **StandardScaler**: Feature scaling/normalization
- **One-Hot Encoding**: Categorical variable transformation
- **IQR Method**: Outlier detection and treatment
- **Log Transformation**: Target variable normalization

## Installation & Setup

### Prerequisites

- Python 3.12 or higher
- pip or uv package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-repo/real-estate-price-prediction.git
   cd real-estate-price-prediction
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

5. **Run the main script** (optional)
   ```bash
   python main.py
   ```

## Workflow & Methodology

### Phase 1: Data Extraction & Validation

**Notebook**: `01_data_extraction.ipynb`

- Load dataset from CSV
- Verify schema and data types
- Identify unique identifiers (`Order`, `PID`)
- Generate metadata summary (27 features with missing values)
- Cross-check with data dictionary

**Key Findings**:
- Dataset shape: 2930 rows × 82 columns
- 43 categorical features (object)
- 28 integer features
- 11 float features
- No duplicate rows found

### Phase 2: Preprocessing & Exploratory Data Analysis

**Notebook**: `02_preprocessing_eda.ipynb`

#### Data Cleaning

1. **Dropped columns with >80% missing values**: `Pool QC`, `Misc Feature`, `Alley`, `Fence`
2. **Dropped low-variance features**: `Street`, `Utilities`, `Condition 2`, `Roof Matl`, `Heating`, `Land Slope`
3. **Imputed missing values**:
   - Categorical: Filled with "None" for basement/garage features
   - Numerical: Filled with 0 for area/count features
   - `Lot Frontage`: Filled with neighborhood-wise median
   - `Electrical`: Filled with mode

**Result**: Cleaned dataset with 2930 rows × 72 columns

#### Exploratory Data Analysis

**Univariate Analysis**:
- Target variable (`SalePrice`) is right-skewed → requires log transformation
- Most numerical features are right-skewed (lot area, living area, basement size)
- Quality ratings (`Overall Qual`) are slightly left-skewed (more high-quality homes)

**Bivariate Analysis**:
- **Top correlations with SalePrice**:
  - Overall Qual: 0.80 (strongest predictor)
  - Gr Liv Area: 0.71
  - Garage Cars: 0.65
  - Garage Area: 0.64
  - Total Bsmt SF: 0.63
  - 1st Flr SF: 0.62

**Multivariate Analysis**:
- Correlation heatmap reveals multicollinearity between size-related features
- Quality features show strong, consistent relationships with price
- Temporal features (Year Built, Year Remod) show moderate positive correlation

#### Outlier Detection

Using **IQR method** (1.5 × IQR rule):
- SalePrice: 137 outliers (luxury properties)
- Gr Liv Area: 75 outliers (large homes)
- Lot Area: 127 outliers (oversized lots)
- Total Bsmt SF: 124 outliers

**Decision**: Outliers capped using IQR bounds rather than removed (to preserve information)

### Phase 3: Feature Engineering

**Notebook**: `03_feature_engineering.ipynb` / Main notebook

#### New Features Created

1. **Age-related features**:
   - `Age_at_Sale` = `Yr Sold` - `Year Built`
   - `Remod_Age_at_Sale` = `Yr Sold` - `Year Remod/Add`
   - `Is_Remodeled` = Binary flag (1 if remodeled)
   - `Garage_Age_at_Sale` = Age of garage at sale time

2. **Area aggregation features**:
   - `Total_SF` = `Total Bsmt SF` + `1st Flr SF` + `2nd Flr SF` (correlation: 0.78)
   - `Total_Bath` = `Bsmt Full Bath` + 0.5×`Bsmt Half Bath` + `Full Bath` + 0.5×`Half Bath`
   - `Total_Porch_SF` = Sum of all porch/deck areas

#### Target Variable Transformation

- Applied `log1p()` transformation to create `SalePrice_Log`
- Resulted in near-normal distribution (better for linear models)

#### Feature Scaling & Encoding

- **Numerical features (40 features)**: Scaled using `StandardScaler` (mean=0, std=1)
- **Categorical features (43 features)**: One-hot encoded with `drop_first=True`

**Final Dataset**: 2930 rows × 270 features (ready for modeling)

### Phase 4: Modeling

**Notebook**: `04_modeling.ipynb`

Models to be implemented:
- Linear Regression (baseline)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- ElasticNet
- Random Forest Regressor
- Gradient Boosting (XGBoost/LightGBM)

Evaluation metrics:
- R² Score (coefficient of determination)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Cross-validation scores

### Phase 5: Visualization & Storytelling

**Notebook**: `05_visualization_storytelling.ipynb`

- Model performance comparison
- Feature importance analysis
- Residual plots
- Prediction vs Actual plots
- Business insights and recommendations

## Key Results & Insights

### Data Insights

1. **Quality Over Size**: Overall quality (0.80 correlation) is more important than raw size metrics
2. **Neighborhood Matters**: NoRidge, StoneBr, and NridgHt have highest average prices ($320K+)
3. **Temporal Trends**: Newer homes and recently remodeled properties command premium prices
4. **Feature Engineering Impact**: `Total_SF` (0.78) outperforms individual area features

### Feature Importance (Top 10)

| Rank | Feature | Correlation | Type |
|------|---------|-------------|------|
| 1 | Overall Qual | 0.80 | Quality Rating |
| 2 | Total_SF | 0.78 | Engineered (Area) |
| 3 | Gr Liv Area | 0.71 | Physical Size |
| 4 | Total_Bath | 0.67 | Engineered (Count) |
| 5 | Garage Cars | 0.65 | Capacity |
| 6 | Garage Area | 0.64 | Physical Size |
| 7 | Total Bsmt SF | 0.63 | Physical Size |
| 8 | 1st Flr SF | 0.62 | Physical Size |
| 9 | Year Built | 0.56 | Temporal |
| 10 | Full Bath | 0.55 | Count |

## Courses & Learning Outcomes

This project integrates concepts from the following BITS Pilani courses:

1. **Statistical Modelling & Inferencing**: Correlation analysis, hypothesis testing, distribution analysis
2. **Data Pre-processing**: Missing value imputation, outlier treatment, feature scaling
3. **Feature Engineering**: Creating derived features, transformations, encoding strategies
4. **Data Visualization & Storytelling**: EDA visualizations, heatmaps, distribution plots
5. **Data Stores & Pipelines**: Data loading, transformation pipelines, file management

## Usage Examples

### Loading the Processed Data

```python
import pandas as pd

# Load original data
df_original = pd.read_csv('data/AmesHousing.csv')

# Load cleaned data
df_cleaned = pd.read_csv('data/AmesHousing_cleaned.csv')

# Load engineered data (ready for modeling)
df_engineered = pd.read_csv('data/AmesHousing_engineered.csv')

# Separate features and target
X = df_engineered.drop(columns=['SalePrice_Log'])
y = df_engineered['SalePrice_Log']
```

### Quick EDA

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Correlation with target
correlations = df_cleaned.corr()['SalePrice'].sort_values(ascending=False)
print(correlations.head(10))

# Visualize top correlations
top_features = correlations.head(11).index
sns.heatmap(df_cleaned[top_features].corr(), annot=True, cmap='coolwarm')
plt.show()
```

### Feature Engineering Example

```python
# Calculate house age
df['Age_at_Sale'] = df['Yr Sold'] - df['Year Built']

# Create total square footage
df['Total_SF'] = df['Total Bsmt SF'] + df['1st Flr SF'] + df['2nd Flr SF']

# Total bathrooms (weighted)
df['Total_Bath'] = (df['Bsmt Full Bath'] +
                    0.5 * df['Bsmt Half Bath'] +
                    df['Full Bath'] +
                    0.5 * df['Half Bath'])
```

## Contributing

This is an academic project. For suggestions or improvements:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

This project is for educational purposes as part of the Advanced Apex Project at BITS Pilani Digital.

## Acknowledgments

- **Dataset**: Ames Housing Dataset (Dean De Cock)
- **Institution**: BITS Pilani Digital
- **Course**: Advanced Apex Project 1 (First Trimester 2025-26)
- **Team**: The Outliers

## Contact & Support

For questions or issues related to this project:

- Check the comprehensive notebook: `real_estate_price_prediction.ipynb`
- Review phase-wise notebooks in the `notebooks/` directory
- Refer to the schema summary: `docs/schema_summary.csv`

---

**Project Status**: In Progress (Phases 1-3 Complete, Phases 4-5 In Development)

**Last Updated**: October 2025
