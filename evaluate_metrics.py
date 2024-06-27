import numpy as np

def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = [line.strip().split(',') for line in lines]
    return data[1:]

def clean_data(data):
    cleaned_data = []
    for row in data:
        cleaned_row = [
            item.replace("'", "").replace('"', '').replace('[', '').replace(']', '') for item in row]
        cleaned_data.append(cleaned_row)
    return cleaned_data

def calculate_metrics(data):
    data = np.array(data)

    # Extract metrics (starting from the 5th column to the end)
    metrics = data[:, 4:].astype(float)

    # Calculate averages, standard deviations, and standard errors
    averages = np.mean(metrics, axis=0)
    standard_deviations = np.std(metrics, axis=0)
    errors = standard_deviations / np.sqrt(metrics.shape[0])
    return averages, standard_deviations, errors

def main(file_path):
    data = read_data_from_file(file_path)
    data = clean_data(data)

    averages, standard_deviations, errors = calculate_metrics(data)

    metric_names = ['AUC', 'AP', 'SN', 'SP', 'ACC', 'PREC', 'REC', 'F1', 'MCC']

    # Create the LaTeX formatted output
    latex_output = "Baseline & " + " & ".join(
        [f"{avg:.2f} $\\pm$ {sd:.2f}" for avg, sd in zip(averages, standard_deviations)]
    ) + " \\\\"

    print(latex_output)

if __name__ == "__main__":
    input_file_path = 'data/eval_results/baseline_output.txt'
    print(f"\nRunning the script with {input_file_path} as the input file:")
    main(input_file_path)