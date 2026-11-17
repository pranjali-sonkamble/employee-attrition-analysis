# Data

The raw dataset is not stored in this repository.

## Download Instructions

1. Download **IBM HR Analytics Employee Attrition & Performance** from Kaggle:
   https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

   **Kaggle download note:** a Kaggle account/sign-in may be required to download the dataset. The repository does not redistribute the raw CSV.

2. Save the CSV file as:

   `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv`

3. From the project root, run:

   `python src/inspect_data.py`

4. Verify that the inspection reports 1,470 rows and 35 columns.

## Data Handling

The raw CSV is excluded from Git using `.gitignore`. Review the source's licence terms before redistributing the dataset.

The dataset is described by its Kaggle uploader as fictional data created by IBM data scientists. Analysis results should not be represented as findings about IBM's actual workforce.
