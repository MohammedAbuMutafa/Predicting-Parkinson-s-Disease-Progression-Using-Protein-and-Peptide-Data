# Import libraries
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
proteins_data = pd.read_csv("train_proteins.csv")
peptides_data = pd.read_csv("train_peptides.csv")
clinical_data = pd.read_csv("train_clinical_data.csv")


# Print missing values percentage in clinical_data
print(clinical_data.isnull().sum() / clinical_data.shape[0] * 100)
print(proteins_data.isnull().sum())
print(peptides_data.isnull().sum())
print(clinical_data.isnull().sum())

# Unique patients
PATS = clinical_data.patient_id.unique()
print('There are', len(PATS), 'unique patients')

# Handle missing values in 'updrs_1', 'updrs_2', 'updrs_3'
def fill_missing_updrs_values():
    clinical_data['updrs_1'] = clinical_data['updrs_1'].fillna(clinical_data['updrs_1'].mean())
    clinical_data['updrs_2'] = clinical_data['updrs_2'].fillna(clinical_data['updrs_2'].mean())
    clinical_data['updrs_3'] = clinical_data['updrs_3'].fillna(clinical_data['updrs_3'].mean())

# Function to fill missing values using linear regression
def fill_missing_upd23b_using_regression():
    # Select relevant columns
    selected_columns = ['updrs_1', 'updrs_2', 'updrs_3', 'upd23b_clinical_state_on_medication']

    # Convert 'On' to 1 and 'Off' to 0 for all values
    clinical_data['upd23b_clinical_state_on_medication'] = clinical_data['upd23b_clinical_state_on_medication'].apply(lambda x: 1 if x == 'On' else (0 if x == 'Off' else x))

    # Split the data into rows with missing and non-missing values
    missing_values = clinical_data[selected_columns][clinical_data['upd23b_clinical_state_on_medication'].isnull()]
    non_missing_values = clinical_data[selected_columns].dropna()

    # Split the data into features (X) and target variable (y) for non-missing values
    X = non_missing_values[['updrs_1', 'updrs_2', 'updrs_3']]
    y = non_missing_values['upd23b_clinical_state_on_medication']

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model for non-missing values
    model.fit(X, y)

    # Convert missing values to NaN and fill using the linear regression model
    missing_values['upd23b_clinical_state_on_medication'] = np.round(model.predict(missing_values[['updrs_1', 'updrs_2', 'updrs_3']]))
    clinical_data.loc[clinical_data['upd23b_clinical_state_on_medication'].isnull(), 'upd23b_clinical_state_on_medication'] = missing_values['upd23b_clinical_state_on_medication']

# Call the functions to fill missing values
fill_missing_updrs_values()
fill_missing_upd23b_using_regression()

# Verify the changes
print(clinical_data.isnull().sum())
clinical_data.to_csv("sample1.csv")



#check if there is a relationship between upd23b and NPX: (no relationship)
#Merge clinical_data and proteins_data on 'visit_id' and 'patient_id'
# merged_data = pd.merge(clinical_data, proteins_data, on=['visit_id', 'patient_id'])

# # # Select relevant columns for analysis
# selected_columns = ['updrs_1', 'updrs_2', 'updrs_3', 'upd23b_On', 'upd23b_Off', 'NPX']

# # # Drop rows where any of the selected columns is null
# df_filtered = merged_data[selected_columns].dropna()

# # # Scatter plot
# sns.pairplot(df_filtered, hue='upd23b_On', markers=['o', 's', 'D'], palette='Set2', vars=['updrs_1', 'updrs_2', 'updrs_3', 'NPX'])
# plt.show()



#check if there is a relationship between updrs values and upd23b 
# Select relevant columns
selected_columns = ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4', 'upd23b_clinical_state_on_medication']

# Drop rows where any of the selected columns is null
df_filtered = clinical_data[selected_columns].dropna()

# # Scatter plot
sns.pairplot(df_filtered, hue='upd23b_clinical_state_on_medication', markers=['o', 's', 'D'], palette='Set2')
plt.show()


# Feature Scaling
numerical_cols = ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']
scaler = StandardScaler()
clinical_data[numerical_cols] = scaler.fit_transform(clinical_data[numerical_cols])



# Visualize histograms before and after scaling
plt.figure(figsize=(18, 5))

# Plot histograms before scaling
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(1, 4, i)  # Change 3 to 4
    plt.hist(clinical_data[col], bins=20, color='blue', alpha=0.7)
    plt.title(f'Histogram of {col} before scaling')

plt.tight_layout()
plt.show()

# Plot histograms after scaling
plt.figure(figsize=(18, 5))

for i, col in enumerate(numerical_cols, 1):
    plt.subplot(1, 4, i)  # Change 3 to 4
    plt.hist(clinical_data[col], bins=20, color='orange', alpha=0.7)
    plt.title(f'Histogram of {col} after scaling')

plt.tight_layout()
plt.show()


# Group by 'patient_id' and aggregate values
aggregated_data = clinical_data.groupby('patient_id').agg({
    'visit_id': 'first',
    'visit_month': 'sum',
    'updrs_1': 'mean',
    'updrs_2': 'mean',
    'updrs_3': 'mean',
    'updrs_4': 'mean',
    'upd23b_clinical_state_on_medication': 'first'
}).reset_index()

# Save the aggregated data to a new CSV file
aggregated_data.to_csv("aggregated_sample.csv", index=False)