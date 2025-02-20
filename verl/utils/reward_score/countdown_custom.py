import re
import ast
import operator


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        pass

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    print("\n\nMatches\n\n", matches)
    if matches:
        final_answer = matches[0].group(1).strip()
        if "=" in final_answer:
            final_answer = final_answer.split("=")[0].strip()
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [abs(int(n)) for n in re.findall(r'\d+', equation_str)]
        
        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        
        # Each number number used should used at most once  
        return all(available_numbers.count(n) >= numbers_in_eq.count(n) for n in numbers_in_eq)
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    target = ground_truth['target']
    numbers = ground_truth['numbers']

    print("\n\n\n\n--------------------------------")
    # Remove all <|endoftext|>
    solution_str = solution_str.replaceAll("<|endoftext|>", "")
    print("Solution str", solution_str)
    print("--------------------------------")
    print("Target", target)
    print("--------------------------------")
    print("Numbers", numbers)
    print("--------------------------------")
    
    equation = extract_solution(solution_str=solution_str)
    print("--------------------------------")
    print("Equation", equation)
    print("--------------------------------")
    do_print = True
    
    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0
    
    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print("Eqauation", equation)
            print("Numbers", numbers)
            print(f"Invalid equation")
        return format_score
        
    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score
            
        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"\n\n\n\nCorrect equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score 


def test():
    
    # Test 1: Correct answer
    solution_str = "<answer> 10 + 22 - 9 </answer>"
    ground_truth = {
        "target": 23,
        "numbers": [10, 22, 9]
    }
    assert compute_score(solution_str, ground_truth) == 1.0

    # Test 2: Include negative numbers
    solution_str = "<answer>-10 - 22</answer>"
    ground_truth = {
        "target": -32,
        "numbers": [10, 22, -9]
    }
    assert compute_score(solution_str, ground_truth) == 1.0

    # Test 3: Include parentheses
    solution_str = "<answer>(10 + 22) - 9</answer>"
    ground_truth = {
        "target": 23,
        "numbers": [10, 22, 9]
    }
    assert compute_score(solution_str, ground_truth) == 1.0

    # Test 4: Fails if uses an illegal number
    solution_str = "<answer>10 + 22 - 9</answer>"
    ground_truth = {
        "target": 23,
        "numbers": [10, 22]
    }
    assert compute_score(solution_str, ground_truth) != 1.0

    # Test 5: Fails if doesn't hit the target
    solution_str = "<answer>10 + 22 + 9</answer>"
    ground_truth = {
        "target": 24,
        "numbers": [10, 22]
    }
    assert compute_score(solution_str, ground_truth) != 1.0




if __name__ == "__main__":
    test()