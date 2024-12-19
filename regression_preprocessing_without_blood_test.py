# Import libraries
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
proteins_data = pd.read_csv("./data/train_proteins.csv")
peptides_data = pd.read_csv("./data/train_peptides.csv")
clinical_data = pd.read_csv("./data/train_clinical_data.csv")
supplement = pd.read_csv("./data/supplemental_clinical_data.csv")

# Print missing values percentage in clinical_data
print(clinical_data.isnull().sum() / clinical_data.shape[0] * 100)

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



#check if there is a relationship between updrs values and upd23b 
# Select relevant columns
selected_columns = ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4', 'upd23b_clinical_state_on_medication']

# Drop rows where any of the selected columns is null
df_filtered = clinical_data[selected_columns].dropna()










train_1= clinical_data[["patient_id","visit_month","updrs_1","upd23b_clinical_state_on_medication"]]
train_2= clinical_data[["patient_id","visit_month","updrs_2","upd23b_clinical_state_on_medication"]]
train_3= clinical_data[["patient_id","visit_month","updrs_3","upd23b_clinical_state_on_medication"]]


# Create a new DataFrame with the total sum of months visited for each patient
total_months_visited = train_1.groupby('patient_id')['visit_month'].sum().reset_index()
total_months_visited.columns = ['patient_id', 'total_months_visited']

# Create a new DataFrame with the highest month visited for each patient
max_month_visited = train_1.groupby('patient_id')['visit_month'].max().reset_index()
max_month_visited.columns = ['patient_id', 'max_month_visited']

# Merge the max_month_visited DataFrame with each of the existing DataFrames
train_1 = pd.merge(train_1, max_month_visited, on='patient_id', how='left')
train_2 = pd.merge(train_2, max_month_visited, on='patient_id', how='left')
train_3 = pd.merge(train_3, max_month_visited, on='patient_id', how='left')



def get_updrs_change(updrs_first, updrs_last):
    return updrs_last - updrs_first

    


# Apply the function to calculate UPDRS change for each patient
train_1['updrs_1'] = train_1.groupby('patient_id')['updrs_1'].transform(lambda x: get_updrs_change(x.iloc[0], x.iloc[-1]))
train_2['updrs_2'] = train_2.groupby('patient_id')['updrs_2'].transform(lambda x: get_updrs_change(x.iloc[0], x.iloc[-1]))
train_3['updrs_3'] = train_3.groupby('patient_id')['updrs_3'].transform(lambda x: get_updrs_change(x.iloc[0], x.iloc[-1]))

# Calculate the difference in visit month between consecutive visits for each patient
clinical_data['visit_month_diff'] = clinical_data.groupby('patient_id')['visit_month'].diff()

# Calculate the mean change in visit month for each patient
mean_visit_month_change = clinical_data.groupby('patient_id')['visit_month_diff'].mean().reset_index()
mean_visit_month_change.columns = ['patient_id', 'mean_visit_month_change']

# Merge the mean_visit_month_change DataFrame with each of the existing DataFrames
train_1 = pd.merge(train_1, mean_visit_month_change, on='patient_id', how='left')
train_2 = pd.merge(train_2, mean_visit_month_change, on='patient_id', how='left')
train_3 = pd.merge(train_3, mean_visit_month_change, on='patient_id', how='left')

# Create a new column indicating whether the patient is a frequent visitor
# You can define a threshold for what is considered a frequent visitor, for example, 4
threshold = 6
train_1['is_frequent_visitor'] = train_1['mean_visit_month_change'].apply(lambda x: 1 if x <= threshold else 0)
train_2['is_frequent_visitor'] = train_2['mean_visit_month_change'].apply(lambda x: 1 if x <= threshold else 0)
train_3['is_frequent_visitor'] = train_3['mean_visit_month_change'].apply(lambda x: 1 if x <= threshold else 0)

train_1['mean_visit_month_change'] = np.round(train_1['mean_visit_month_change']).astype(int)
train_2['mean_visit_month_change'] = np.round(train_2['mean_visit_month_change']).astype(int)
train_3['mean_visit_month_change'] = np.round(train_3['mean_visit_month_change']).astype(int)

# Function to replace 'upd23b_clinical_state_on_medication' with most frequent value for each patient
def replace_upd23b_with_most_frequent(df):
    most_frequent_upd23b = df.groupby('patient_id')['upd23b_clinical_state_on_medication'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).reset_index()
    most_frequent_upd23b.columns = ['patient_id', 'most_frequent_upd23b']
    
    df = pd.merge(df, most_frequent_upd23b, on='patient_id', how='left')
    df['upd23b_clinical_state_on_medication'] = df['most_frequent_upd23b'].fillna(df['upd23b_clinical_state_on_medication'])
    df = df.drop(['most_frequent_upd23b'], axis=1)
    
    return df

# Replace 'upd23b_clinical_state_on_medication' with most frequent value for each patient in each train set
train_1 = replace_upd23b_with_most_frequent(train_1)
train_2 = replace_upd23b_with_most_frequent(train_2)
train_3 = replace_upd23b_with_most_frequent(train_3)



# # Calculate the mean, min, max, and std for PeptideAbundance for each unique patient ID
# peptide_summary = peptides_data.groupby('patient_id')['PeptideAbundance'].agg(['mean', 'min', 'max', 'std']).reset_index()
# peptide_summary.columns = ['patient_id', 'PeptideAbundance_mean', 'PeptideAbundance_min', 'PeptideAbundance_max', 'PeptideAbundance_std']


# # Calculate the mean, min, max, and std for NPX for each unique patient ID
# protein_summary = proteins_data.groupby('patient_id')['NPX'].agg(['mean', 'min', 'max', 'std']).reset_index()
# protein_summary.columns = ['patient_id', 'NPX_mean', 'NPX_min', 'NPX_max', 'NPX_std']




# # Merge mean and summary values
# mean_values = pd.merge(peptide_summary, protein_summary, on='patient_id', how='outer')
# train_1 = pd.merge(train_1, mean_values, on='patient_id', how='left')
# train_2 = pd.merge(train_2, mean_values, on='patient_id', how='left')
# train_3 = pd.merge(train_3, mean_values, on='patient_id', how='left')



# mms = MinMaxScaler()
# scale_col = ['NPX_min', 'NPX_max', 'NPX_mean', 'NPX_std', 'PeptideAbundance_min', 'PeptideAbundance_max', 'PeptideAbundance_mean', 'PeptideAbundance_std']
# train_1[scale_col] = mms.fit_transform(train_1[scale_col])
# train_2[scale_col] = mms.fit_transform(train_2[scale_col])
# train_3[scale_col] = mms.fit_transform(train_3[scale_col])


# Drop 'visit_id' and 'updrs_1' columns
train_1 = train_1.drop(['visit_month'], axis=1)
train_2 = train_2.drop(['visit_month'], axis=1)
train_3 = train_3.drop(['visit_month'], axis=1)



# Remove duplicate rows
train_1 = train_1.drop_duplicates()
train_2 = train_2.drop_duplicates()
train_3 = train_3.drop_duplicates()



# Feature Scaling
scaler = StandardScaler()


train_1.to_csv('train_1.csv')
train_2.to_csv('train_2.csv')
train_3.to_csv('train_3.csv')
