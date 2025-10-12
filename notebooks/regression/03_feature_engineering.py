# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Feature Engineering
# ## Feature Engineering Goals
# 1.  **Categorical Feature Encoding**: Convert nominal and ordinal categorical features into numerical representations.
#     - **Ordinal**: `age`, `education`, `experience_overall_years`, `experience_field_years`.
#     - **Nominal (Low Cardinality)**: `gender`.
#     - **Nominal (High Cardinality/Text)**: `industry`, `job`, `race`. These require careful handling to extract meaningful information.
# 2.  **Numerical Feature Engineering**: Create new features through combinations or transformations of existing numerical and ordinal features (e.g., interaction terms, ratios).
# 3.  **Feature Scaling**: Standardize numerical features for models sensitive to feature scales.

# +
# Imports
import json
from pathlib import Path

import polars as pl
from sklearn.preprocessing import StandardScaler
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
# -

# Constants
CLEANED_FILE_PATH: str = "../../data/cleaned/regression"

# Load Feature Sets
output_dir = Path(CLEANED_FILE_PATH)
X_train = pl.read_csv(output_dir / "X_train.csv")
X_val = pl.read_csv(output_dir / "X_val.csv")
X_test = pl.read_csv(output_dir / "X_test.csv")

# ## Ordinal Feature Encoding
# Features like `age`, `education`, and `experience` have a natural order. We will map these categories to numerical ranks to preserve this information.

# +
# Define Ordinal Categories in Order
age_order: list[str] = [
    "under 18",
    "18-24",
    "25-34",
    "35-44",
    "45-54",
    "55-64",
    "65 or over",
]

education_order: list[str] = [
    "High School",
    "Some college",
    "College degree",
    "Professional degree (MD, JD, etc.)",
    "Master's degree",
    "PhD",
]

experience_order: list[str] = [
    "1 year or less",
    "2 - 4 years",
    "5-7 years",
    "8 - 10 years",
    "11 - 20 years",
    "21 - 30 years",
    "31 - 40 years",
    "41 years or more",
]

ordinal_mappings: dict[str, dict[int, str]] = {
    "age": age_order,
    "education": education_order,
    "experience_overall_years": experience_order,
    "experience_field_years": experience_order,
}


# -


def apply_ordinal_encoding(df: pl.DataFrame) -> pl.DataFrame:
    """Applies direct ordinal mapping to specified columns."""
    for key, value in ordinal_mappings.items():
        df = df.with_columns(
            pl.col(key).cast(pl.Enum(value)).cast(pl.Int32).alias(f"{key}_ordinal")
        )

    return df.drop(ordinal_mappings.keys())


# Apply Ordinal Encoding
X_train_encoded = apply_ordinal_encoding(X_train)
X_test_encoded = apply_ordinal_encoding(X_test)
X_val_encoded = apply_ordinal_encoding(X_val)
X_train_encoded

# ## One-Hot Encoding
# Features like `gender`, `race`, and `industry` have no inherent ranking, so One-Hot Encoding is appropriate.
#
# Since gender is a single-choice question, just encode the gender column into its preset choices.

# Polars has a convenient to_dummies() function for One-Hot Encoding
X_train_encoded = X_train_encoded.to_dummies("gender")
X_test_encoded = X_test_encoded.to_dummies("gender")
X_val_encoded = X_val_encoded.to_dummies("gender")
X_train_encoded

# `race` and `industry` are more complicated, as they are multiple-choice responses in the survey.
#
# For our purposes, we will encode the preset choices given in the survey, and filter out the free-text responses to reduce cardinality and decrease encodings.

# +
# List out preset choices
# All commas are removed since they mess with the encoding process
PRESET_RACE_OPTIONS: list[str] = [
    "White",
    "Asian or Asian American",
    "Black or African American",
    "Hispanic Latino or Spanish origin",
    "Middle Eastern or Northern African",
    "Native American or Alaska Native",
    "Another option not listed here or prefer not to answer",
]
PRESET_RACE_OPTIONS.sort()

PRESET_INDUSTRY_OPTIONS: list[str] = [
    "Accounting Banking & Finance",
    "Agriculture or Forestry",
    "Art & Design",
    "Business or Consulting",
    "Computing or Tech",
    "Education (Primary/Secondary)",
    "Education (Higher Education)",
    "Engineering or Manufacturing",
    "Entertainment",
    "Government and Public Administration",
    "Health care",
    "Hospitality & Events",
    "Insurance",
    "Law",
    "Law Enforcement & Security",
    "Leisure Sport & Tourism",
    "Marketing Advertising & PR",
    "Media & Digital",
    "Nonprofits",
    "Property or Construction",
    "Recruitment or HR",
    "Retail",
    "Sales",
    "Social Work",
    "Transport or Logistics",
    "Utilities & Telecommunications",
]


# -

def apply_one_hot_encoding(df: pl.DataFrame) -> pl.DataFrame:
    # Remove all commas
    df = df.with_columns(
        pl.col("industry").str.replace_all(r",", "").alias("industry"),
        pl.col("race").str.replace_all(r",", "").alias("race"),
    )

    # Add Index
    df = df.with_row_index()

    # Industry
    df_industry = (df
        .select(pl.col("index"), pl.col("industry").str.split(","))
        .with_columns(pl.lit(1).alias("__one__"))
        .explode("industry")
        .pivot(index="index", on="industry", values="__one__")
        .select(["index"] + (PRESET_INDUSTRY_OPTIONS))
        .fill_null(0)
        .rename({option: f"industry_{option}" for option in PRESET_INDUSTRY_OPTIONS})
    )

    # Race
    df_race = (df
         .select(pl.col("index"), pl.col("race").str.split(","))
        .with_columns(pl.lit(1).alias("__one__"))
        .explode("race")
        .pivot(index="index", on="race", values="__one__")
        .select(["index"] + (PRESET_RACE_OPTIONS))
        .fill_null(0)
        .rename({option: f"race_{option}" for option in PRESET_RACE_OPTIONS})
        )

    # Drop Raw Columns
    return (df
        .join(df_industry, on="index")
        .join(df_race, on="index")
        .drop(["industry", "race"]))


# +
# Remove all commas and add index
X_train_one_hot = apply_one_hot_encoding(X_train_encoded)
X_test_one_hot = apply_one_hot_encoding(X_test_encoded)
X_val_one_hot = apply_one_hot_encoding(X_val_encoded)

X_train_one_hot
# -

# ## Feature Extraction
# The `job` column is free-text, and therefore has extremely high cardinality.
#
# Let's try extracting what we can from it first, although this really should use NLP instead...

# +
levels: list[str] = [
    "junior",
    "senior",
    "manager",
    "director",
    "vp",
    "c_level",
    "other",
]

def extract_job_features(df: pl.DataFrame) -> pl.DataFrame:
    # Add Index
    df = df.with_columns(
        pl.col("job").str.to_lowercase().fill_null("unknown").alias("job_cleaned")
    )

    # Extract Common Job Levels
    df = df.with_columns(
        pl.when(pl.col("job_cleaned").str.contains("intern|junior|entry"))
        .then(pl.lit("junior"))
        .when(pl.col("job_cleaned").str.contains("senior|lead|principal"))
        .then(pl.lit("senior"))
        .when(pl.col("job_cleaned").str.contains("manager|supervisor"))
        .then(pl.lit("manager"))
        .when(pl.col("job_cleaned").str.contains("director"))
        .then(pl.lit("director"))
        .when(pl.col("job_cleaned").str.contains("vp|vice president"))
        .then(pl.lit("vp"))
        .when(pl.col("job_cleaned").str.contains("cfo|ceo|cto|cmo|coo"))
        .then(pl.lit("c_level"))
        .otherwise(pl.lit("other"))
        .alias("job_level")
    )

    # One-Hot Encoding for extracted features
    df_job = (
        df.select(pl.col("index"), pl.col("job_level").str.split(","))
        .with_columns(pl.lit(1).alias("__one__"))
        .explode("job_level")
        .pivot(index="index", on="job_level", values="__one__")
        .select(["index"] + levels)
        .fill_null(0)
        .rename({option: f"job_level_{option}" for option in levels})
    )

    # Drop raw columns
    return (
        df
            .join(df_job, on="index")
            .drop(["job", "job_cleaned", "job_level"])
    )



# -

X_train_job = extract_job_features(X_train_one_hot)
X_test_job = extract_job_features(X_test_one_hot)
X_val_job = extract_job_features(X_val_one_hot)
X_train_job


# Let's also encode the date the survey response is submitted, just in case it's somehow relevant.

def extract_timestamp(df: pl.DataFrame) -> pl.DataFrame:
    # Convert timestamp string to DateTime object
    df = df.with_columns(
        pl.col("timestamp")
        .str.to_datetime("%m/%d/%Y %H:%M:%S", strict=False)
        .alias("timestamp_dt")
    )

    # Extract Month, Day, Hour
    df = df.with_columns(
        [
            pl.col("timestamp_dt").dt.month().alias("month"),
            pl.col("timestamp_dt").dt.weekday().alias("day_of_week"),  # 1=Monday, 7=Saunday
            pl.col("timestamp_dt").dt.hour().alias("hour"),
        ]
    )

    # Drop Raw Columns
    return df.drop(["timestamp", "timestamp_dt"])



X_train_dt = extract_timestamp(X_train_job)
X_test_dt = extract_timestamp(X_test_job)
X_val_dt = extract_timestamp(X_val_job)
X_train_dt


# ## Numerical Feature Engineering
# Let's also create some composite features like field experience / overall experience.

def create_composite_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        (
            pl.col("experience_field_years_ordinal")
            / pl.col("experience_overall_years_ordinal").replace(0, 1)  # Replace 0 with 1 to avoid division by zero
        )
        .alias("experience_ratio")
        .fill_nan(0.0),
        (pl.col("education_ordinal") * pl.col("experience_field_years_ordinal")).alias(
            "education_experience_interaction"
        ),
        (pl.col("experience_overall_years_ordinal") ** 2).alias("experience_overall_sq"),
    )

    return df


# +
X_train_final_features = create_composite_features(X_train_dt)
X_test_final_features = create_composite_features(X_test_dt)
X_val_final_features = create_composite_features(X_val_dt)

X_train_final_features
# -

# ## Feature Scaling
# Now that we're finally done setting up all our features, we need to scale them to an appropriate range to avoid feature domination.

# +
# Create Scalar
scaler = StandardScaler()

numerical_cols = X_train_final_features.select(
    [
        pl.exclude(
            [
                col
                for col in X_train_final_features.columns
                if X_train_final_features[col].dtype == pl.UInt8 or col == "index"
            ]
        )
    ]
).columns

print(f"\nNumerical features to scale: {numerical_cols}")

scaler = StandardScaler()

# Scale each feature set (and fit training set)
X_train_scaled = pl.DataFrame(
    scaler.fit_transform(X_train_final_features[numerical_cols]), schema=numerical_cols
)
X_test_scaled = pl.DataFrame(
    scaler.transform(X_test_final_features[numerical_cols]), schema=numerical_cols
)
X_val_scaled = pl.DataFrame(
    scaler.transform(X_val_final_features[numerical_cols]), schema=numerical_cols
)

# Save Scalar
scaler_metadata = {
    "features_scaled": numerical_cols,
    "n_samples_fit": len(X_train_final_features),
    "mean": scaler.mean_.tolist(),
    "std": scaler.scale_.tolist(),
}
joblib.dump(scaler, output_dir / "scaler.pkl")
with open(output_dir / "scaler_metadata.json", "w") as f:
    json.dump(scaler_metadata, f, indent=2)
    
# Concat everything back
non_scaled_cols = [
    col for col in X_train_final_features.columns if col not in numerical_cols
]

X_train_final = pl.concat(
    [X_train_scaled, X_train_final_features[non_scaled_cols]], how="horizontal"
).drop("index")
X_test_final = pl.concat(
    [X_test_scaled, X_test_final_features[non_scaled_cols]], how="horizontal"
).drop("index")
X_val_final = pl.concat(
    [X_val_scaled, X_val_final_features[non_scaled_cols]], how="horizontal"
).drop("index")

X_train_final
# -




# Final Check
assert X_train_final.shape[1] == X_val_final.shape[1] == X_test_final.shape[1], \
    "Feature count mismatch across engineered datasets!"
print(f"✓ All engineered datasets have {X_train_final.shape[1]} features.")

# Save the engineered feature sets
X_train_final.write_csv(output_dir / "X_train_engineered.csv")
X_val_final.write_csv(output_dir / "X_val_engineered.csv")
X_test_final.write_csv(output_dir / "X_test_engineered.csv")

# ## Feature-Target Correlation Analysis
# Since we're at the last step before model building, let's check again the correlation between features and the target.

y_train_log = pl.read_csv(output_dir / "y_train_log.csv")

# +
df_with_target = X_train_final.with_columns(
    y_train_log.rename({"annual_salary_log": "target_log_salary"})
)

numerical_cols = df_with_target.select(
    [
        pl.exclude(
            [
                col
                for col in df_with_target.columns
                if df_with_target[col].dtype == pl.UInt8 or col == "index"
            ]
        )
    ]
)

df = (
    pl.DataFrame(
        {
            "Feature": numerical_cols.columns,
            "Correlation": numerical_cols.corr().select(pl.col("target_log_salary"))
        }
    )
    .sort(pl.col("Correlation").abs(), descending=True)
    .head(10)
)

plt.figure(figsize=(24, 8))
sns.catplot(df, x="Feature", y ="Correlation", legend=True, hue="Feature", palette="husl", legend_out=True)

plt.title("Top 10 Feature Correlations with Log-Transformed Annual Salary")
plt.xticks([])
plt.savefig("../../figures/feature_correlation.png", dpi=300)
# -


