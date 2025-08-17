from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score,  recall_score, f1_score
from copy import deepcopy as dc
from class_symmetry_feature_maps_submission import *
import os
import csv
import json
from time import time
from matplotlib import pyplot as plt
import seaborn as sns; sns.set_theme()
import sys
import pandas as pd
from class_quiskit_feature_maps_submission import *

scaling = 1 #0: [0,1], 1: [0,pi], 2: [0,2pi] #zero seems fine, two pi not working
n_sampless = [100, 250, 500, 1000]  # Set this to the number of samples you used to generate the synthetic data

def plot_vectors(vectors, data_name=None, group_type=None, is_3d=False, add=None):
    fig = plt.figure(figsize=(8, 8))
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    colors = ['red', 'green', 'blue', 'black']

    for vec in vectors:
        # Determine the color based on the quadrant
        color_index = 0
        if vec[0] >= 0 and vec[1] >= 0:
            color_index = 0  # First quadrant
        elif vec[0] < 0 and vec[1] >= 0:
            color_index = 1  # Second quadrant
        elif vec[0] < 0 and vec[1] < 0:
            color_index = 2  # Third quadrant
        elif vec[0] >= 0 and vec[1] < 0:
            color_index = 3  # Fourth quadrant

        # Plotting the vector
        if is_3d:
            ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=colors[color_index])
        else:
            ax.quiver(0, 0, vec[0], vec[1], color=colors[color_index])

    if is_3d:
        add = add + "_3d"
    plot_filename_png = f"{kernel_matrix_dir}/{data_name}_{group_type}_kernel_matrix_vectors{add}.png"
    plot_filename_eps = f"{kernel_matrix_dir}/{data_name}_{group_type}_kernel_matrix_vectors{add}.eps"
    plt.title(f'Kernel Matrix Vectors for Training - {data_name} - {group_type}')
    #plt.tight_layout()
    #plt.grid(True)
    plt.savefig(plot_filename_png)
    plt.savefig(plot_filename_eps)
    plt.close()

# Directory for saving kernel matrices
kernel_matrix_dir = 'kernel_matrices_visualizations'
if not os.path.exists(kernel_matrix_dir):
    os.makedirs(kernel_matrix_dir)

def check_and_convert_matrix(matrix):
    """
    Check if the matrix is complex and convert it to a real matrix if it's real.
    """
    if np.all(np.isreal(matrix)):
        print("Kernel matrix is real")
        return matrix.real
    else:
        print("Kernel matrix is complex")
        return matrix

def process_samples(X, feature_map):
    """
    Process the samples using the feature map

    Parameters:
    X (np.ndarray): Input data.
    feature_map (function): Feature map function.

    Returns:
    list: List of transformed samples.
    """
    processed_samples = []
    for x in X:
        processed_samples.append(feature_map(x))
    return processed_samples

def compute_kernel_matrix_train(X, feature_map, return_vectors=True):
    processed_samples = process_samples(X, feature_map)
    num_samples = len(X)

    # Determine the data type based on the output of the feature map (complex or real)
    sample_type = type(processed_samples[0][0])
    dtype = np.complex128 if sample_type == complex else np.float64

    kernel_matrix = np.zeros((num_samples, num_samples), dtype=dtype)

    for i in range(num_samples):
        for j in range(num_samples):
            kernel_value = np.absolute(np.dot(processed_samples[i], processed_samples[j].conj()))**2
            kernel_matrix[i, j] = kernel_value

    if return_vectors:
        return check_and_convert_matrix(kernel_matrix), processed_samples
    return check_and_convert_matrix(kernel_matrix)

def compute_kernel_matrix_test(X_train, X_test, feature_map):
    processed_samples_train = process_samples(X_train, feature_map)
    processed_samples_test = process_samples(X_test, feature_map)

    num_samples_train = len(X_train)
    num_samples_test = len(X_test)

    kernel_matrix_test = np.zeros((num_samples_test, num_samples_train), dtype=np.float64)

    for i in range(num_samples_test):
        for j in range(num_samples_train):
            kernel_value = np.absolute(np.dot(processed_samples_test[i], processed_samples_train[j].conj()))**2
            kernel_matrix_test[i, j] = kernel_value

    return check_and_convert_matrix(kernel_matrix_test)

def calculate_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='macro'),
        "recall": recall_score(y_true, y_pred, average='macro'),
        "f1_score": f1_score(y_true, y_pred, average='macro'),
    }
    return metrics

output_real = False #usually false

add = ""
if output_real:
    add = add + "_output_real"
if scaling == 0:
    add = add + "_scale0"
if scaling == 1:
    add = add + "_scale1"
if scaling == 2:
    add = add + "_scale2"
if scaling == 3:
    add = add + "_scale3"
if scaling == 4:
    add = add + "_scale4"
if scaling == 5:
    add = add + "_scale5"
if scaling == 6:
    add = add + "_scale6"
if scaling == 7:
    add = add + "_scale7"

results_log = []
csv_file_int = f"results_log_int{add}_2024_synth_fin.csv"
json_file_int = f"results_log_int{add}_2024_synth_fin.json"
csv_columns = []

def load_synthetic_dataset(n_samples):
    file_name = f"synthetic_data_{n_samples}.csv"
    try:
        df = pd.read_csv(file_name)
        X = df[['x0', 'x1', 'x2']].values
        y = df['Class'].values
        return X, y, f"Synthetic_{n_samples}"
    except FileNotFoundError:
        print(f"No such file: {file_name}")
        sys.exit()

# Loop through datasets
#for data_nr in range(0, 16):
for data_nr in range(4):
    print('#############################################################################################################')
    print(f"Using dataset number: {data_nr}")
    X, y, data_name = dc(load_synthetic_dataset(n_sampless[data_nr]))

    n_features__ = len(np.array(X)[0,:])
    print(f"Using dataset: {data_name}")
    print(f"Sample Count: {len(y)}")
    print(f"n_features: {n_features__}")

    # Instantiate the SymmetryFeatureMaps class for the current dataset
    symmetry_feature_maps = SymmetryFeatureMaps(X.shape[1])

    # Loop through feature maps (or symmetry groups in this case)
    for group_type in [ 'Q_Z', 'SO', 'SL', 'SU', 'GL', 'U', 'O']:
        # Split the dataset into training and testing sets
        print('performing train/test split')

        X_train, X_test, y_train, y_test = dc(train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True))

        if group_type.startswith("Q_"):

            classic_feature_map_name = dc(group_type[2:])
            print(f'#############################################################################################################')
            print(f'{classic_feature_map_name} - Feature Map')

            group_n = 0
            n_generators = 0
            generators = []
            n_classes = len(np.unique(y_train))
            n_features = len(X_train[0, :])

            print(f'Classes in {data_name}: {n_classes}')
            classic_feature_map = ClassicalFeatureMap()

            # Use the apply_feature_map method of the SymmetryFeatureMaps instance
            feature_map = lambda x: classic_feature_map.apply_feature_map(x, classic_feature_map_name)

            print(f'Feature Map: {feature_map}')

        else:
            print(f'#############################################################################################################')
            print(f'Group Type: {group_type}')
            if group_type == "SO":
                group_n = symmetry_feature_maps.size_SO
                generators = symmetry_feature_maps.group_generators_SO
                n_generators = len(symmetry_feature_maps.group_generators_SO)
            elif group_type == "SL":
                group_n = symmetry_feature_maps.size_SL
                generators = symmetry_feature_maps.group_generators_SL
                n_generators = len(symmetry_feature_maps.group_generators_SL)
            elif group_type == "SU":
                group_n = symmetry_feature_maps.size_SU
                generators = symmetry_feature_maps.group_generators_SU
                n_generators = len(symmetry_feature_maps.group_generators_SU)
            elif group_type == "GL":
                group_n = symmetry_feature_maps.size_GL
                generators = symmetry_feature_maps.group_generators_GL
                n_generators = len(symmetry_feature_maps.group_generators_GL)
            elif group_type == "U":
                group_n = symmetry_feature_maps.size_U
                generators = symmetry_feature_maps.group_generators_U
                n_generators = len(symmetry_feature_maps.group_generators_U)
            elif group_type == "O":
                group_n = symmetry_feature_maps.size_O
                generators = symmetry_feature_maps.group_generators_O
                n_generators = len(symmetry_feature_maps.group_generators_O)
            elif group_type == "T":
                group_n = symmetry_feature_maps.size_T
                generators = symmetry_feature_maps.group_generators_T
                n_generators = len(symmetry_feature_maps.group_generators_T)


            n_classes = len(np.unique(y_train))
            n_features = len(X_train[0,:])

            print(f'Classes in {data_name}: {n_classes}')

            # Use the apply_feature_map method of the SymmetryFeatureMaps instance
            feature_map= lambda x: symmetry_feature_maps.apply_feature_map(x, group_type, output_real=output_real, return_group_n=False)

        print(f'Feature Map: {feature_map}')

        # Compute kernel matrices using the feature map
        start_time = time()
        kernel_matrix_train, processed_samples = compute_kernel_matrix_train(X_train, feature_map)
        print(f'n_features: {n_features}')
        #print(np.shape(processed_samples))
        n_features_groupvector = len(processed_samples[0])
        print(f'group_n: {group_n}')
        print(f'n_features_groupvector: {n_features_groupvector}')
        print(f'n_generators: {n_generators}')


        if n_generators < n_features:
            print('Here is an error: n_generators < n_features')
            print(f"n_generators: {n_generators}")
            print(f"generators: {generators}")
            print(f"n_features: {n_features}")
            print(f"features: {X_train[0, :]}")


        kernel_matrix_train_time = time() - start_time
        start_time = time()
        kernel_matrix_test = compute_kernel_matrix_test(X_train, X_test, feature_map)
        kernel_matrix_test_time = time() - start_time

        kernel_matrix_csv_filename = f"{kernel_matrix_dir}/{data_name}_{group_type}_kernel_matrix{add}.csv"
        #np.savetxt(kernel_matrix_csv_filename, kernel_matrix_train, delimiter=",")

        # Plot and save the kernel matrix
        threshold = np.percentile(kernel_matrix_train, 90)  # e.g., 99th percentile cipping
        kernel_matrix_train = dc(np.clip(kernel_matrix_train, None, threshold))

        plot_filename_png = f"{kernel_matrix_dir}/{data_name}_{group_type}_kernel_matrix{add}.png"
        plot_filename_eps = f"{kernel_matrix_dir}/{data_name}_{group_type}_kernel_matrix{add}.eps"
        plt.figure(figsize=(20, 16))
        sns.heatmap(kernel_matrix_train, annot=False, cmap='viridis')
        plt.title(f'Kernel Matrix for Training - {data_name} - {group_type}')
        plt.savefig(plot_filename_png)
        #plt.savefig(plot_filename_eps)
        plt.close()

        print('Kernel matrices calculated, running SVM')
        # Train and predict with SVM
        svm = SVC(kernel='precomputed')
        start_time = time()
        svm.fit(kernel_matrix_train, y_train)
        svm_training_time = time() - start_time
        start_time = time()
        svm_predictions = svm.predict(kernel_matrix_test)
        svm_prediction_time = time() - start_time
        svm_metrics = calculate_metrics(y_test, svm_predictions)

        # Evaluate accuracy
        svm_accuracy = accuracy_score(y_test, svm_predictions)
        print(f"Accuracy with feature map {feature_map.__name__}: {svm_accuracy}")

        # Initialize and train CatBoost Classifier
        catboost_model = CatBoostClassifier(verbose=0)  # verbose=0 to avoid lengthy output
        start_time = time()
        catboost_model.fit(X_train, y_train)
        catboost_training_time = time() - start_time

        # Predict and evaluate
        start_time = time()
        catboost_predictions = catboost_model.predict(X_test)
        catboost_prediction_time = time() - start_time
        catboost_accuracy = accuracy_score(y_test, catboost_predictions)
        catboost_metrics = calculate_metrics(y_test, catboost_predictions)
        print(f"CatBoost Accuracy: {catboost_accuracy}")

        # Log results
        result_entry = {
            "dataset": data_name,
            "n_features": n_features,
            "n_group_features": n_features_groupvector,
            "group_type": group_type,
            "group_n": group_n,
            "n_generators": n_generators,
            "svm_accuracy": svm_metrics["accuracy"],
            "svm_precision": svm_metrics["precision"],
            "svm_recall": svm_metrics["recall"],
            "svm_f1_score": svm_metrics["f1_score"],
            "svm_training_time": svm_training_time,
            "svm_prediction_time": svm_prediction_time,
            "svm_kernel_matrix_time": kernel_matrix_train_time,
            "catboost_accuracy": catboost_metrics["accuracy"],
            "catboost_precision": catboost_metrics["precision"],
            "catboost_recall": catboost_metrics["recall"],
            "catboost_f1_score": catboost_metrics["f1_score"],
            "catboost_training_time": catboost_training_time,
            "catboost_prediction_time": catboost_prediction_time
        }
        results_log.append(result_entry)

        if not csv_columns:
            csv_columns = list(result_entry.keys())

        # Save intermediate results to CSV
        try:
            with open(csv_file_int, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                if data_nr == 11 and group_type == 'SO':  # Write header only at the beginning
                    writer.writeheader()
                writer.writerow(result_entry)
        except IOError:
            print("I/O error in CSV writing")

        # Save intermediate results to JSON
        try:
            with open(json_file_int, 'w') as jsonfile:
                json.dump(results_log, jsonfile, indent=4)
        except IOError:
            print("I/O error in JSON writing")

# Save to CSV
csv_columns = list(results_log[0].keys())
csv_file = f"results_log{add}_2024_synth_fin.csv"
try:
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in results_log:
            writer.writerow(data)
except IOError:
    print("I/O error in CSV writing")

# Save to JSON
json_file = f"results_log{add}_2024_synth_fin.json"
with open(json_file, 'w') as jsonfile:
    json.dump(results_log, jsonfile, indent=4)


