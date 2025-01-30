import json
from typing import List, Callable, Tuple
from random import randint, seed
import random

from data_labels import TRAIN, TEST, OFF_TARGET, OFF_AVAILABLE_NUMS, OFF_POSITIVE_NUMBERS, OFF_SIX_AVAILABLE, OFF_SEVEN_AVAILABLE, OFF_EIGHT_AVAILABLE

def generate_data(
    num_samples: int,
    num_numbers_range: Tuple[int, int],
    numbers_range: Tuple[int, int],
    # If exclusion callback is provided,
    # the generated data will not contain
    # any samples that satisfy the callback
    exclusion_callback: Callable[[int, List[int]], bool] = None,
):
    samples = []
    num_excluded = 0

    while len(samples) < num_samples:
        test_data = generate_test_data(num_numbers_range, numbers_range)
        if exclusion_callback is not None and exclusion_callback(test_data):
            num_excluded += 1
            continue
        samples.append(test_data)

    return samples

def generate_test_data(
    num_numbers_range: Tuple[int, int],
    numbers_range: Tuple[int, int],
):
    """
    Generate a single test data sample, where you are guaranteed 
    to be able to have a to reach the target numbers using
    the permitted operations.
    """
    
    # How many numbers are available?
    num_numbers = randint(num_numbers_range[0], num_numbers_range[1])

    # Generate the numbers available (guaranteed unique)
    numbers_available = random.sample(range(numbers_range[0], numbers_range[1] + 1), num_numbers)

    # Select between 2 and len(numbers) numbers to be used to make the target
    num_numbers_to_use = random.randint(2, len(numbers_available))
    numbers_to_use = random.sample(numbers_available, num_numbers_to_use)

    # Randomly flip the sign of some of the numbers
    equation_parts = []
    for i in range(len(numbers_to_use)):
        if random.random() < 0.4:
            equation_parts.append(-numbers_to_use[i])
        else:
            equation_parts.append(numbers_to_use[i])
    # Generate the target
    target = sum(equation_parts)

    return {
        "numbers_available": numbers_available,
        "numbers_to_use": numbers_to_use,
        "target": target,
        "equation_parts": equation_parts,
    }

BASE_NUM_NUMBERS_RANGE = (3, 5)
BASE_NUMBERS_RANGE = (1, 60)
BASE_SEED_VALUE = 1

def generate_on_domain_data(num_samples: int, seed_value: int):

    def exclusion_callback(test_data):
        # Exclue if the target is between 60 and 64
        if 60 <= test_data["target"] <= 68:
            print(f"Excluding -- target {test_data['target']}")
            return True

        # Exclude if any of the numbers are 40, 41
        if any(number in [40] for number in test_data["numbers_available"]):
            print(f"Excluding -- numbers_available {test_data['numbers_available']}")
            return True

        # Exclude if you are only adding numbers, and there are > 2 numbers
        if all(part > 0 for part in test_data["equation_parts"]) and len(test_data["numbers_to_use"]) > 3:
            print(f"Excluding -- too positive {test_data['equation_parts']}")
            return True

        return False

    return generate_data(num_samples, BASE_NUM_NUMBERS_RANGE, BASE_NUMBERS_RANGE, exclusion_callback)


def generate_off_domain_data_target(num_samples: int, seed_value: int):
    def exclusion_callback(test_data):
        if test_data["target"] < 60 or test_data["target"] > 68:
            return True
        return False
    return generate_data(num_samples, BASE_NUM_NUMBERS_RANGE, BASE_NUMBERS_RANGE, exclusion_callback)

def generate_off_domain_data_numbers(num_samples: int, seed_value: int):
    def exclusion_callback(test_data):
        return not any(number in [40] for number in test_data["numbers_available"])
    return generate_data(num_samples, BASE_NUM_NUMBERS_RANGE, BASE_NUMBERS_RANGE, exclusion_callback)


def generate_off_domain_data_positive_numbers(num_samples: int, seed_value: int):
    def exclusion_callback(test_data):
        return not (all(part > 0 for part in test_data["equation_parts"]) and len(test_data["numbers_to_use"]) > 3)
    return generate_data(num_samples, BASE_NUM_NUMBERS_RANGE, BASE_NUMBERS_RANGE, exclusion_callback)

if __name__ == "__main__":
    seed(1)

    """
    Generate 200,000 samples from the on domain
    data, which will be used to train the model
    """
    on_domain_data = generate_on_domain_data(200000, 1)
    for data in on_domain_data:
        assert data["target"] < 60 or data["target"] > 68
        assert not any(number in [40] for number in data["numbers_available"])
        assert not (all(part > 0 for part in data["equation_parts"]) and len(data["numbers_to_use"]) > 3)

    """
    Generate 100 samples for EACH off domain category
    """
    # Target being between 60 and 68
    off_domain_target_data = generate_off_domain_data_target(100, 1)
    for data in off_domain_target_data:
        assert data["target"] >= 60 and data["target"] <= 68

    # Numbers available containing 40
    off_domain_available_nums_data = generate_off_domain_data_numbers(100, 1)
    for data in off_domain_available_nums_data:
        assert any(number in [40] for number in data["numbers_available"])

    # All numbers are positive in solution, and > 3 numbers are used
    off_domain_positive_numbers_data = generate_off_domain_data_positive_numbers(100, 1)
    for data in off_domain_positive_numbers_data:
        assert all(part > 0 for part in data["equation_parts"]) and len(data["numbers_to_use"]) > 3

    # Numbers available being 6, 7, or 8
    off_domain_six_available = generate_data(100, (6, 6), BASE_NUMBERS_RANGE, None)
    off_domain_seven_available = generate_data(100, (7, 7), BASE_NUMBERS_RANGE, None)
    off_domain_eight_available = generate_data(100, (8, 8), BASE_NUMBERS_RANGE, None)

    #
    """
    Join all the data together, with labels:
    - TRAIN: 190,000
    - TEST: 10,000
    - OFF_DOMAIN_TARGET: 100
    - OFF_DOMAIN_AVAILABLE_NUMS: 100
    - OFF_DOMAIN_POSITIVE_NUMBERS: 100
    - OFF_DOMAIN_SIX_AVAILABLE: 100
    - OFF_DOMAIN_SEVEN_AVAILABLE: 100
    - OFF_DOMAIN_EIGHT_AVAILABLE: 100
    """

    on_domain_data = [{"label": TRAIN, **data} for data in on_domain_data[:190000]]
    test_data = [{"label": TEST, **data} for data in on_domain_data[190000:]]
    off_domain_target_data = [{"label": OFF_TARGET, **data} for data in off_domain_target_data]
    off_domain_available_nums_data = [{"label": OFF_AVAILABLE_NUMS, **data} for data in off_domain_available_nums_data]
    off_domain_positive_numbers_data = [{"label": OFF_POSITIVE_NUMBERS, **data} for data in off_domain_positive_numbers_data]
    off_domain_six_available = [{"label": OFF_SIX_AVAILABLE, **data} for data in off_domain_six_available]
    off_domain_seven_available = [{"label": OFF_SEVEN_AVAILABLE, **data} for data in off_domain_seven_available]
    off_domain_eight_available = [{"label": OFF_EIGHT_AVAILABLE, **data} for data in off_domain_eight_available]

    datas = on_domain_data + test_data + off_domain_target_data + off_domain_available_nums_data + off_domain_positive_numbers_data + off_domain_six_available + off_domain_seven_available + off_domain_eight_available

    with open("data.json", "w") as f:
        json.dump(datas, f, indent=4)

    #print(json.dumps(off_domain_available_nums_data, indent=4))


    #print(json.dumps(datas, indent=4))
