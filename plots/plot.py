import numpy as np


def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = [line.strip().split(', ') for line in lines]
    return data


def clean_data(data):
    cleaned_data = []
    for row in data:
        cleaned_row = [
            item.replace("'", "").replace('"', '').replace('[', '').replace(
                ']', '') for item in row]
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
    # auc_score, ap_score, sn, sp, acc, prec, rec, f1, mcc
    metric_names = ['AUC', 'AP', 'SN', 'SP', 'ACC', 'PREC', 'REC', 'F1', 'MCC']

    # Create the output row
    output_row = ['Average'] + averages.round(2).tolist() + [
        'SD'] + standard_deviations.round(2).tolist() + [
                     'Error'] + errors.round(2).tolist()

    print(', '.join(['Metric'] + metric_names + ['Metric'] + metric_names + [
        'Metric'] + metric_names))
    print(', '.join(map(str, output_row)))


if __name__ == "__main__":
    print(f"Running the script with 'MPM.txt' as the input file:")
    input_file_path = 'MPM.txt'
    main(input_file_path)


    print("\nRunning the script with 'hybrid.txt' as the input file:")

    input_file_path = 'hybrid.txt'
    main(input_file_path)