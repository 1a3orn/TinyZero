# Implementing
# https://verl.readthedocs.io/en/latest/preparation/prepare_data.html
#
#
# So, specifically making a data format that is compatible with verl
# where each bit of data has the fields
# - data_source (string)
# - prompt (a list of dicts with role and content)
# - ability (string)
# - reward_model (dict with style and ground_truth)
# - extra_info (dict with split and index)

# All this is saved as a parquet file.

import json
import pandas as pd
import argparse
from data_instance import DataInstance
from data_labels import DATA_SOURCE, TRAIN, TEST, OFF_TARGET, OFF_AVAILABLE_NUMS, OFF_POSITIVE_NUMBERS, OFF_SIX_AVAILABLE, OFF_SEVEN_AVAILABLE, OFF_EIGHT_AVAILABLE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file_prefix", type=str, required=True)
    args = parser.parse_args()  
    # Load the input file which we assume will be a json
    # with the following fields:
    # - label: TRAIN | TEST | OFF_TARGET | OFF_AVAILABLE_NUMS | OFF_POSITIVE_NUMBERS | OFF_SIX_AVAILABLE | OFF_SEVEN_AVAILABLE | OFF_EIGHT_AVAILABLE
    # - numbers_available: list of numbers
    # - numbers_to_use: list of numbers
    # - target: int
    # - equation_parts: list of numbers

    with open(args.input_file, 'r') as f:
        data = json.load(f)

    def make_map_fn(data):
        d = DataInstance(data)
        return {
            "data_source": DATA_SOURCE,
            "prompt": d.question_text_messages(),
            "prompt_text": d.question_text_base(),
            "ability": "countdown",
            "reward_model": {
                "style": "countdown",
                "ground_truth": {
                    "target": data["target"],
                    "numbers": data["numbers_available"]
                }
            }
        }

    # Define labels and their corresponding file suffixes
    label_map = {
        TRAIN: "train",
        TEST: "train",
        OFF_TARGET: "off_target",
        OFF_AVAILABLE_NUMS: "off_available_nums",
        OFF_POSITIVE_NUMBERS: "off_positive_numbers",
        OFF_SIX_AVAILABLE: "off_six_available",
        OFF_SEVEN_AVAILABLE: "off_seven_available",
        OFF_EIGHT_AVAILABLE: "off_eight_available"
    }

    # Create dataframes and save to parquet in one loop
    for label, suffix in label_map.items():
        print(f"Processing {label} with suffix {suffix}")
        data_filtered = [make_map_fn(d) for d in data if label.lower() in d["label"].lower()]
        # If data is not train, limit to 50
        if label == TRAIN:
            data_filtered = data_filtered[:180000]
        elif label == TEST:
            # Peel test from train
            data_filtered = [make_map_fn(d) for d in data if d["label"] == TRAIN]
            data_filtered = data_filtered[180000:180050]
        else:
            data_filtered = data_filtered[:50]
        print(f"Filtered data length: {len(data_filtered)}")
        df = pd.DataFrame(data_filtered)
        df.to_parquet(f"{args.output_file_prefix}_{suffix}.parquet")


if __name__ == "__main__":
    main()

