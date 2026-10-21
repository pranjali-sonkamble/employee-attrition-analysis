
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df = pd.read_csv(DATA_PATH)

print("COLUMN VALIDATION")
print("=" * 50)

print("\nIdentifier check: EmployeeNumber")
print("nunique:", df["EmployeeNumber"].nunique(), "of", len(df))
print("min/max:", df["EmployeeNumber"].min(), df["EmployeeNumber"].max())

print("\nConstant columns:")
for col in ["EmployeeCount", "Over18", "StandardHours"]:
    print(f"{col}: {df[col].unique().tolist()}")

print("\nCoded ordinal candidates (value counts, sorted by value):")
ordinal_cols = [
    "Education", "EnvironmentSatisfaction", "JobInvolvement",
    "JobSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance",
    "PerformanceRating", "StockOptionLevel", "JobLevel",
]
for col in ordinal_cols:
    print(f"\n{col}:")
    print(df[col].value_counts().sort_index().to_string())

print("\nNumeric summary:")
numeric_cols = [
    "Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyIncome",
    "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike",
    "TotalWorkingYears", "TrainingTimesLastYear", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]
print(df[numeric_cols].describe().T.round(2).to_string())

print("\nLogical consistency checks (count of violating rows):")
print("YearsAtCompany > TotalWorkingYears:",
      (df["YearsAtCompany"] > df["TotalWorkingYears"]).sum())
print("YearsInCurrentRole > YearsAtCompany:",
      (df["YearsInCurrentRole"] > df["YearsAtCompany"]).sum())
print("YearsWithCurrManager > YearsAtCompany:",
      (df["YearsWithCurrManager"] > df["YearsAtCompany"]).sum())
print("YearsSinceLastPromotion > YearsAtCompany:",
      (df["YearsSinceLastPromotion"] > df["YearsAtCompany"]).sum())
print("Age - TotalWorkingYears < 14 (started work under 14):",
      (df["Age"] - df["TotalWorkingYears"] < 14).sum())
