import json
import csv
from os.path import realpath as realpath


def json_to_csv(path_to_json_file, path_to_csv_file=None):
    real_path_to_json_file = realpath(path_to_json_file)
    real_path_to_csv_file = realpath(path_to_csv_file)

    with open(real_path_to_json_file, "r") as json_file:
        # Load the json file
        data = json.load(json_file)

    # Open the CSV file
    if not real_path_to_csv_file:
        real_path_to_csv_file = data.csv
    with open(real_path_to_csv_file, "w", newline="") as csv_file:
        # Create a CSV writer
        writer = csv.writer(csv_file)

        # Write the CSV header
        writer.writerow(data[0].keys())

        # Write the CSV data
        for item in data:
            writer.writerow(item.values())


def csv_to_json(path_to_csv_file, path_to_json_file=None):
    real_path_to_json_file = realpath(path_to_json_file)
    real_path_to_csv_file = realpath(path_to_csv_file)
    with open(real_path_to_csv_file, "r") as csv_file:
        # Create a new CSV reader
        reader = csv.reader(csv_file)

        # Skip the header row
        header = next(reader)

        # Initialise an empty list for the JSON data
        data = []

        # Read each row in the CSV file
        for row in reader:
            # Create a dictionary for each row
            item = {}
            for i in range(len(header)):
                item[header[i]] = row[i]
            # Add this dictionary to the data list
            data.append(item)

    # Open the JSON file
    if not real_path_to_json_file:
        real_path_to_json_file = data.json
    with open(real_path_to_json_file, "w") as json_file:
        # Write the JSON data
        json.dump(data, json_file, indent=4)
