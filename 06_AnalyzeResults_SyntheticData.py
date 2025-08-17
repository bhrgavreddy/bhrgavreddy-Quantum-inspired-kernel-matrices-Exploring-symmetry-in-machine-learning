import json
import numpy as np
import pandas as pd

# Correcting the path and reloading the JSON data
json_path = "results_log_scale1_2024_synth_fin.json"
print(json_path)
# Load JSON data from the file
with open(json_path, 'r') as file:
    json_data = json.load(file)

# Convert JSON data to a pandas DataFrame
df = pd.json_normalize(json_data)

print(json_path)
print("accuracy")
# Since we're taking the first CatBoost result per dataset, we'll extract those
df_catboost_first_per_dataset_accuracy = df.groupby('dataset').first().reset_index()[['dataset', 'catboost_accuracy']]

# Pivoting the SVM data to have group types as columns
df_svm_pivot_accuracy = df.pivot_table(index='dataset', columns='group_type', values='svm_accuracy', aggfunc='first').reset_index()

# Merging the pivoted SVM dataframe with the first CatBoost result per dataset
df_final_accuracy = pd.merge(df_svm_pivot_accuracy, df_catboost_first_per_dataset_accuracy, on='dataset', how='left')

# Dynamically creating the LaTeX table headers based on unique group types
group_types = df['group_type'].unique()
group_headers = " & ".join([f"{gt}" for gt in group_types])
latex_header = f"data set $\\downarrow$ \\textbackslash approach & {group_headers} & CatBoost \\\\"

# Dynamically constructing the rows of the LaTeX table
rows_latex_accuracy = "\n".join(df_final_accuracy.apply(lambda x: " & ".join([x['dataset']] + [f"{x[gt]:.4f}" if gt in x else 'N/A' for gt in group_types] + [f"{x['catboost_accuracy']:.4f}"] ), axis=1) + "\\\\")

# Complete LaTeX table code
latex_code_final_accuracy = f"""
\\begin{{table}}[]
\\begin{{tabular}}{{{'l' + 'c' * (len(group_types) + 1)}}}
{latex_header} \\ toprule
{rows_latex_accuracy}
\\end{{tabular}}
\\end{{table}}
"""

print(latex_code_final_accuracy)






print(json_path)
print("precision")
# Since we're taking the first CatBoost result per dataset, we'll extract those
df_catboost_first_per_dataset_precision = df.groupby('dataset').first().reset_index()[['dataset', 'catboost_precision']]

# Pivoting the SVM data to have group types as columns
df_svm_pivot_precision = df.pivot_table(index='dataset', columns='group_type', values='svm_precision', aggfunc='first').reset_index()

# Merging the pivoted SVM dataframe with the first CatBoost result per dataset
df_final_precision = pd.merge(df_svm_pivot_precision, df_catboost_first_per_dataset_precision, on='dataset', how='left')

# Dynamically creating the LaTeX table headers based on unique group types
group_types = df['group_type'].unique()
group_headers = " & ".join([f"{gt}" for gt in group_types])
latex_header = f"data set $\\downarrow$ \\textbackslash approach & {group_headers} & CatBoost \\\\"

# Dynamically constructing the rows of the LaTeX table
rows_latex_precision = "\n".join(df_final_precision.apply(lambda x: " & ".join([x['dataset']] + [f"{x[gt]:.4f}" if gt in x else 'N/A' for gt in group_types] + [f"{x['catboost_precision']:.4f}"] ), axis=1) + "\\\\")

# Complete LaTeX table code
latex_code_final_precision = f"""
\\begin{{table}}[]
\\begin{{tabular}}{{{'l' + 'c' * (len(group_types) + 1)}}}
{latex_header} \\ toprule
{rows_latex_precision}
\\end{{tabular}}
\\end{{table}}
"""

print(latex_code_final_precision)








print(json_path)
print("recall")
# Since we're taking the first CatBoost result per dataset, we'll extract those
df_catboost_first_per_dataset_recall = df.groupby('dataset').first().reset_index()[['dataset', 'catboost_recall']]

# Pivoting the SVM data to have group types as columns
df_svm_pivot_recall = df.pivot_table(index='dataset', columns='group_type', values='svm_recall', aggfunc='first').reset_index()

# Merging the pivoted SVM dataframe with the first CatBoost result per dataset
df_final_recall = pd.merge(df_svm_pivot_recall, df_catboost_first_per_dataset_recall, on='dataset', how='left')

# Dynamically creating the LaTeX table headers based on unique group types
group_types = df['group_type'].unique()
group_headers = " & ".join([f"{gt}" for gt in group_types])
latex_header = f"data set $\\downarrow$ \\textbackslash approach & {group_headers} & CatBoost \\\\"

# Dynamically constructing the rows of the LaTeX table
rows_latex_recall = "\n".join(df_final_recall.apply(lambda x: " & ".join([x['dataset']] + [f"{x[gt]:.4f}" if gt in x else 'N/A' for gt in group_types] + [f"{x['catboost_recall']:.4f}"] ), axis=1) + "\\\\")

# Complete LaTeX table code
latex_code_final_recall = f"""
\\begin{{table}}[]
\\begin{{tabular}}{{{'l' + 'c' * (len(group_types) + 1)}}}
{latex_header} \\ toprule
{rows_latex_recall}
\\end{{tabular}}
\\end{{table}}
"""

print(latex_code_final_recall)



print(json_path)
print("f1_score")
# Since we're taking the first CatBoost result per dataset, we'll extract those
df_catboost_first_per_dataset_f1_score = df.groupby('dataset').first().reset_index()[['dataset', 'catboost_f1_score']]

# Pivoting the SVM data to have group types as columns
df_svm_pivot_f1_score = df.pivot_table(index='dataset', columns='group_type', values='svm_f1_score', aggfunc='first').reset_index()

# Merging the pivoted SVM dataframe with the first CatBoost result per dataset
df_final_f1_score = pd.merge(df_svm_pivot_f1_score, df_catboost_first_per_dataset_f1_score, on='dataset', how='left')

# Dynamically creating the LaTeX table headers based on unique group types
group_types = df['group_type'].unique()
group_headers = " & ".join([f"{gt}" for gt in group_types])
latex_header = f"data set $\\downarrow$ \\textbackslash approach & {group_headers} & CatBoost \\\\"

# Dynamically constructing the rows of the LaTeX table
rows_latex_f1_score = "\n".join(df_final_f1_score.apply(lambda x: " & ".join([x['dataset']] + [f"{x[gt]:.4f}" if gt in x else 'N/A' for gt in group_types] + [f"{x['catboost_f1_score']:.4f}"] ), axis=1) + "\\\\")

# Complete LaTeX table code
latex_code_final_f1_score = f"""
\\begin{{table}}[]
\\begin{{tabular}}{{{'l' + 'c' * (len(group_types) + 1)}}}
{latex_header} \\ toprule
{rows_latex_f1_score}
\\end{{tabular}}
\\end{{table}}
"""

print(latex_code_final_f1_score)
