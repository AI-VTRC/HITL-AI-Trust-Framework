import pandas as pd
import numpy as np
import pymannkendall as mk
from scipy.stats import ttest_ind
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import seaborn as sns
import matplotlib.pyplot as plt

# Load all datasets
datasets = {
    'Control': pd.read_csv(r'C:\Users\install\Documents\HITL-AI-Trust-Framework\Simulation_Results\Original_Algorithm\csv\control_threshold_0.6_originalAlgo.csv'),
    'Sample_1': pd.read_csv(r'C:\Users\install\Documents\HITL-AI-Trust-Framework\Simulation_Results\HITL_Algorithm\csv\Sample_1_HITL.csv'),
    'Sample_2': pd.read_csv(r'C:\Users\install\Documents\HITL-AI-Trust-Framework\Simulation_Results\HITL_Algorithm\csv\Sample_2_HITL.csv'),
    'Sample_3': pd.read_csv(r'C:\Users\install\Documents\HITL-AI-Trust-Framework\Simulation_Results\HITL_Algorithm\csv\Sample_3_HITL.csv'),
    'Sample_4': pd.read_csv(r'C:\Users\install\Documents\HITL-AI-Trust-Framework\Simulation_Results\HITL_Algorithm\csv\Sample_4_HITL.csv'),
    'Sample_5': pd.read_csv(r'C:\Users\install\Documents\HITL-AI-Trust-Framework\Simulation_Results\HITL_Algorithm\csv\Sample_5_HITL.csv'),
    'Orig_1': pd.read_csv(r'C:\Users\install\Documents\HITL-AI-Trust-Framework\Simulation_Results\Original_Algorithm\csv\Sample1_threshold_0.6_originalAlgo.csv'),
    'Orig_2': pd.read_csv(r'C:\Users\install\Documents\HITL-AI-Trust-Framework\Simulation_Results\Original_Algorithm\csv\Sample2_threshold_0.6_originalAlgo.csv'),
    'Orig_3': pd.read_csv(r'C:\Users\install\Documents\HITL-AI-Trust-Framework\Simulation_Results\Original_Algorithm\csv\Sample3_threshold_0.6_originalAlgo.csv'),
    'Orig_4': pd.read_csv(r'C:\Users\install\Documents\HITL-AI-Trust-Framework\Simulation_Results\Original_Algorithm\csv\Sample4_threshold_0.6_originalAlgo.csv'),
    'Orig_5': pd.read_csv(r'C:\Users\install\Documents\HITL-AI-Trust-Framework\Simulation_Results\Original_Algorithm\csv\Sample5_threshold_0.6_originalAlgo.csv')
}

# Initialize an empty DataFrame to store results from all experiments
all_results = pd.DataFrame(columns=['Dataset', 'CAV Receiver', 'CAV Sender', 'Tau'])
all_results['Tau'] = all_results['Tau'].astype(float)

# Process each dataset
for dataset_name, df in datasets.items():
    # Group by CAV Receiver and CAV Sender
    grouped = df.groupby(['CAV Reciever', 'CAV Sender']) # typo in original tables

    # Analyze each group
    results = {}
    for name, group in grouped:
        # Sort by Image Frame Index to ensure the series is in the correct order
        time_series = group.sort_values('Image Frame Index')['Trust Score'].values
        # Perform Mann-Kendall test
        result = mk.original_test(time_series)
        results[name] = result.Tau  # Store the Tau value for each pair

    # Convert results to a DataFrame
    temp_results = pd.DataFrame(list(results.items()), columns=['CAV Pair', 'Tau'])
    temp_results[['CAV Receiver', 'CAV Sender']] = pd.DataFrame(temp_results['CAV Pair'].tolist(),
                                                                index=temp_results.index)
    temp_results.drop(columns=['CAV Pair'], inplace=True)
    temp_results['Dataset'] = dataset_name  # Add the dataset name to each row

    # Drop columns with all NaN values before concatenation
    temp_results.dropna(axis=1, how='all', inplace=True)

    # Concatenate to the main results DataFrame
    all_results = pd.concat([all_results, temp_results], ignore_index=True)

all_results.to_csv(r'C:\Users\install\Documents\HITL-AI-Trust-Framework\Statistics_Analysis\trend_calculation.csv', index=False)

# Load experiment parameters
exp_design = pd.read_excel(r'C:\Users\install\Documents\HITL-AI-Trust-Framework\Statistics_Analysis\randomized_design_colors.xlsx')

# Mapping
exp_design['Dataset'] = exp_design['Experiment (Dataset)'].map({1: 'Sample_1', 2: 'Sample_2', 3: 'Sample_3',
                                                                4: 'Sample_4',  5: 'Sample_5'})
exp_design['CAV Receiver'] = exp_design['User/CAV Assigned'].map({1: 'cav1', 2: 'cav2', 3: 'cav3', 4: 'cav4'})

# Merging on Dataset and CAV Receiver
merged_df = pd.merge(all_results, exp_design, on=['Dataset', 'CAV Receiver'], how='left')

# Drop useless columns
merged_df.drop(columns=['Experiment (Dataset)', 'User/CAV Assigned'], inplace=True)

# Conditionally updating rows
orig_datasets = ['Orig_1', 'Orig_2', 'Orig_3', 'Orig_4', 'Orig_5']
merged_df.loc[merged_df['Dataset'].isin(orig_datasets), 'User_Type'] = 'User_0.6'
merged_df.loc[merged_df['Dataset'].isin(orig_datasets), 'Trust History Required'] = False
merged_df.loc[merged_df['Dataset'].isin(orig_datasets), 'Number of Trust Frames Required'] = 0

control_datasets = ['Control']
merged_df.loc[merged_df['Dataset'].isin(control_datasets), 'User_Type'] = 'User_0.6'
merged_df.loc[merged_df['Dataset'].isin(control_datasets), 'Trust History Required'] = False
merged_df.loc[merged_df['Dataset'].isin(control_datasets), 'Number of Trust Frames Required'] = 0

def assign_experiment_type(dataset):
    if dataset in orig_datasets:
        return 'Orig Trust Framework'
    elif dataset in control_datasets:
        return 'Orig TF on Control Datasets'
    else:
        return 'PerceptiSync Framework'

# Apply the function to the 'Dataset' column to create the 'Exp' column
merged_df['Exp'] = merged_df['Dataset'].apply(assign_experiment_type)

# Create Trust Monitor Flag
merged_df['Trust Monitor'] = merged_df['User_Type'].apply(lambda x: 'TRUE' if x == "User_M" else 'FALSE')

user_type_mapping = {
    'User_0.3': 'Trusting',
    'User_0.6': 'Moderate',
    'User_0.8': 'Cautious',
    'User_M': 'Dynamic'
}

# Replace the user type labels in the DataFrame
merged_df['User_Type'] = merged_df['User_Type'].map(user_type_mapping)
merged_df.to_csv(r'C:\Users\install\Documents\HITL-AI-Trust-Framework\Statistics_Analysis\trend_statistics.csv', index=False)

###############################################
###############################################
# Research Question 1: Original Trust Framework on Control (12) vs Benchmark Datasets (60)
# Create a new DataFrame from a conditionally filtered original DataFrame
orig_df = merged_df[merged_df['Exp'] != 'PerceptiSync Framework'].copy()
orig_df.loc[:, 'Exp'] = orig_df['Exp'].replace('Orig Trust Framework', 'Orig TF on Benchmark Datasets')

def perform_two_tailed_ttest(df, column='Tau', group_column='Exp', alpha=0.05):
    """
    Perform a two-tailed t-test to compare two groups in a dataframe.

    Parameters:
    - df: pandas DataFrame containing the data.
    - column: String, name of the column with numeric data to test.
    - group_column: String, name of the column with group labels.
    - alpha: Float, significance level for deciding on the hypothesis.

    Prints:
    - T-statistic, p-value, group means, group medians, standard deviations, and hypothesis test conclusion.
    """

    # Filter the data into two groups based on the group_column
    group_control = df[df[group_column] == 'Orig TF on Control Datasets'][column]
    group_benchmark = df[df[group_column] == 'Orig TF on Benchmark Datasets'][column]

    # Calculate means and standard deviations
    mean_control = group_control.mean()
    std_control = group_control.std()
    median_control = group_control.median()
    mean_benchmark = group_benchmark.mean()
    std_benchmark = group_benchmark.std()
    median_benchmark = group_benchmark.median()

    # Perform the two-sample t-test (Welch's t-test for unequal variances)
    t_stat, p_value = ttest_ind(group_benchmark, group_control, equal_var=False)

    # Print detailed results
    print("Two-tailed t-test:")
    print(f"Group 'Orig' mean: {mean_control:.3f}, std deviation: {std_control:.3f}")
    print(f"Group 'New' mean: {mean_benchmark:.3f}, std deviation: {std_benchmark:.3f}")
    print(f"Group 'New' median: {median_benchmark:.3f}, Group 'Old' median: {median_control:.3f}")
    print(f"T-statistic: {t_stat:.3f}")
    print(f"Two-tailed p-value: {p_value:.3g}")

    # Decision based on the alpha level
    if p_value < alpha:
        print("We reject the null hypothesis: There is a significant difference in the mean Tau values between 'Orig TF on Benchmark Datasets' and 'Orig TF on Control Datasets'.")
    else:
        print("We fail to reject the null hypothesis: There is no significant difference in the mean Tau values between 'Orig TF on Benchmark Datasets' and 'Orig TF on Control Datasets'.")

# Call the function with the DataFrame
perform_two_tailed_ttest(orig_df)


# Create the boxplot
plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 16})  # Adjust this value as needed
boxplot = sns.boxplot(x='Exp', y='Tau', data=orig_df)
plt.title('Kendall Tau Coefficient of the Original Trust Frame \n by Control and Benchmark datasets')
plt.xlabel('')
plt.ylabel('Kendall Tau Coefficient')
plt.show()

# Research Question 2: Original TF (60) vs PerceptiSync Framework (60) (on benchmark datasets)
# Create the boxplot
plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 16})  # Adjust this value as needed
boxplot = sns.boxplot(x='Exp', y='Tau', data=merged_df)
plt.title('Kendall Tau Coefficient by Framework')
plt.xlabel('')
plt.ylabel('Kendall Tau Coefficient')
plt.show()

def perform_one_tailed_ttest(df, column='Tau', group_column='Exp', alpha=0.05):
    """
    Perform a one-tailed t-test to compare two groups in a dataframe.

    Parameters:
    - dataframe: pandas DataFrame containing the data.
    - column: String, name of the column with numeric data to test.
    - group_column: String, name of the column with group labels.
    - alpha: Float, significance level for deciding on the hypothesis.

    Prints:
    - T-statistic, p-value, group means, group medians, standard deviations, and hypothesis test conclusion.
    """

    # Filter the data into two groups based on the group_column
    group_orig = df[df[group_column] == 'Orig Trust Framework'][column]
    group_new = df[df[group_column] == 'PerceptiSync Framework'][column]

    # Calculate means and standard deviations
    mean_orig = group_orig.mean()
    std_orig = group_orig.std()
    median_orig = group_orig.median()
    mean_new = group_new.mean()
    std_new = group_new.std()
    median_new = group_new.median()

    # Perform the two-sample t-test (Welch's t-test for unequal variances)
    t_stat, p_value = ttest_ind(group_new, group_orig, equal_var=False)

    # Halve the p-value for a one-tailed test if the mean of 'New' is greater than 'Orig'
    if mean_new > mean_orig:
        p_value_one_tailed = p_value / 2
    else:
        p_value_one_tailed = 1 - (p_value / 2)

    # Print detailed results
    print("One-tailed t-test (testing if New > Orig):")
    print(f"Group 'Orig' mean: {mean_orig:.3f}, std deviation: {std_orig:.3f}")
    print(f"Group 'New' mean: {mean_new:.3f}, std deviation: {std_new:.3f}")
    print(f"Group 'New' median: {median_new:.3f}, Group 'Old' median: {median_orig:.3f}")
    print(f"T-statistic: {t_stat:.3f}")
    print(f"One-tailed p-value: {p_value_one_tailed:.3g}")

    # Decision based on the alpha level
    if p_value_one_tailed < alpha:
        print("We reject the null hypothesis: The mean Tau value for 'New' is significantly greater than for 'Orig'.")
    else:
        print("We fail to reject the null hypothesis: There is not enough evidence to conclude that 'New' is greater than 'Orig'.")


perform_one_tailed_ttest(merged_df)

# Research Question 3: PerceptiSync Framework Trust Monitor enabled (15) vs disabled (45) (on benchmark datasets)
new_df = merged_df[merged_df['Exp'] == 'PerceptiSync Framework']

# Visualization of Tau by User_Type
plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 16})  # Adjust this value as needed
sns.boxplot(x='User_Type', y='Tau', data=new_df)
plt.title('Kendall Tau Coefficient by Trust Levels')
plt.ylabel('Kendall Tau Coefficient')
plt.xlabel('')
plt.show()


# Visualization of Tau by Trust History Required
plt.figure(figsize=(10, 6))
sns.boxplot(x='Trust History Required', y='Tau', data=new_df)
plt.title('Tau by Trust History Required')
plt.show()

model = ols('Tau ~ C(User_Type) + C(Q("Trust History Required")) + C(Q("Number of Trust Frames Required")) + C(Dataset)', data=new_df).fit()
anova_results = sm.stats.anova_lm(model, typ=2)
print(anova_results)

tukey = pairwise_tukeyhsd(endog=new_df['Tau'],     # Data
                          groups=new_df['User_Type'],   # Groups
                          alpha=0.05)          # Significance level

# Print the results
print(tukey.summary())