import json
import re

def load_data(path, filter_label=None):
    with open(path, "r") as f:
        data = json.load(f)
    if filter_label is not None:
        data = [d for d in data if d["label"] == filter_label]
    return data

class DataInstance:
    def __init__(self, data):
        self.data = data

    def __str__(self):
        return f"Instance(data={self.data})"

    def system_prompt(self):
        first = "A conversation between User and Assistant. The user asks a question, and the assistant solves it. "
        second = "The assistant first thinks about the reasoning process in its mind and then provides the user with the answer."
        return first + second

    def user_question(self):
        third = f"Using the numbers {self.data['numbers_available']}, create an equation that equals {self.data['target']}. "
        fourth = "You can use operations + and -, i.e, addition, subtraction, and making a number negative. "
        fifth = "Each number may be used once, or not at all. "
        sixth = "Show your work in <think> . . . </think> tags. Return the final answer in <answer>...</answer> tags, "
        seventh = f"for example <answer>10 + 22 - 9</answer> or <answer> 43 - 3 - 4 + 8</answer>."
        return third + fourth + fifth + sixth + seventh

    def question_text_base(self):
        first = self.system_prompt()
        second = "\nUser: " + self.user_question()
        third = "\nAssistant: Let me solve this step by step. <think> "
        return first + second + third

    def question_text_instruct_openai(self):
        messages = [
            {"role": "developer", "content": self.system_prompt()},
            {"role": "user", "content": self.user_question()}
        ]
        return messages

    def is_answer_correct(self, answer: str):
        
        # Peel out contents of <answer> </answer> tags,
        # if they are there.
        if "<answer>" in answer:
            answer = answer.split("<answer>")[1]
        if "</answer>" in answer:
            answer = answer.split("</answer>")[0]
        try:
            total = eval(answer)
            print("TOTAL: ", total)
            print("TARGET: ", self.data["target"])
            sums_correctly = total == self.data["target"]
            
            if "=" in answer:
                answer_split = answer.split("=")[0]

                # Assign answer to whichever was longer of answer_split and answer
                answer = answer_split[0] if len(answer_split[0]) > len(answer_split[1]) else answer_split[1]

            # Use regex to peel out all numbers, negative numbers, decimal numbers.
            numbers_used = [float(x) for x in re.findall(r'-?\d*\.?\d+', answer)]

            # make negative numbers positive
            numbers_used = [abs(x) for x in numbers_used]
            # Check if all numbers used are in the list of numbers available.
            numbers_available = set(self.data["numbers_available"])
            numbers_used = set(numbers_used)
            numbers_used_correctly = numbers_used.issubset(numbers_available)
            return numbers_used_correctly
        except Exception as e:
            return False


test_extract = """<think>To solve this problem, we need to find a combination of the numbers 7, 17, 58, and 47 using only addition and subtraction that equals -115.

1. Start by considering the largest number, 58. If we make it negative, we have -58.
2. Next, consider the second largest number, 47. Making it negative gives us -47.
3. Adding these two negative numbers: -58 - 47 = -105.
4. We are currently at -105, and we need to reach -115.
5. The difference between -105 and -115 is -10.
6. We have the numbers 7 and 17 left. 
7. If we make 17 negative, we have -17.
8. Adding -17 to -105 gives us -122, which is too low.
9. If we make 7 negative, we have -7.
10. Adding -7 to -105 gives us -112, which is still not -115.
11. Let's adjust the approach: Try using 17 positively and 7 negatively.
12. Start with -58 - 47 = -105.
13. Add 17 to -105: -105 + 17 = -88.
14. Subtract 7 from -88: -88 - 7 = -95.
15. The numbers do not add up correctly, so let's try a different combination.
16. Consider using 58 positively and 47 negatively.
17. Start with 58, then subtract 47: 58 - 47 = 11.
18. Subtract 17 from 11: 11 - 17 = -6.
19. Subtract 7 from -6: -6 - 7 = -13.
20. This doesn't work either, so let's try another combination.
21. Let's try using 58, 47, and 17 all negatively.
22. Start with -58 - 47 - 17 = -122.
23. Add 7 to -122: -122 + 7 = -115.

Finally, we have found a combination that works: -58 - 47 - 17 + 7 = -115.</think>

<answer>-58 - 47 - 17 + 7</answer>"""

if __name__ == "__main__":

    print(DataInstance({
        "numbers_available": [7, 17, 58, 47],
        "target": -115,
        "equation_parts": [-58, -47, -17, 7],
    }).is_answer_correct("<answer>-58 - 47 - 17 + 7</answer>"))

    data = load_data("data.json", "TRAIN")
    for d in data[:2]:
        print(DataInstance(d).question_text())
        print(DataInstance(d).is_answer_correct("<answer>37 + 55</answer>"))
        print(DataInstance(d).is_answer_correct("<answer>37.0 + 55</answer>"))
        print(DataInstance(d).is_answer_correct("<answer>37.0 + 55.0</answer>"))
        print(DataInstance(d).is_answer_correct("<answer>37.0 + 55.0</answer>"))
        print("-" * 100)
