
import json
import os
import openai
import time

from data_load import load_data, DataInstance

def create_sft_data(path, filter_label=None):
    data = load_data(path, filter_label)
    print(len(data))
    data_to_save = []
    for d in data[24000:28000]:
        try:
            instance = DataInstance(d)
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            attempt = 0
            while attempt < 4:
                time.sleep(1)
                messages = instance.question_text_instruct_openai()
                messages[1]["content"] += " Be concise."

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.8,
                    max_tokens=1500,
                )
                response_text = response.choices[0].message.content
                is_correct = instance.is_answer_correct(response_text)
                if is_correct:
                    break
                attempt += 1
                print("ATTEMPT: ", attempt)
            print("\n\n\n\n\n")
            print('-' * 100)
            print("RESPONSE: ", response_text)
            print("IS CORRECT: ", is_correct)
            print("TARGET: ", d["target"])
            print("ALL NUMBERS: ", d["numbers_available"])
            print("COULD USE", d["equation_parts"])
            data_to_save.append({
                **d,
                "response": response_text,
                "is_correct": is_correct,
            })
            print(len(data_to_save))
        except Exception as e:
            print(e)
            print('-' * 100)
        
        # Every 500 samples, save the data
        if len(data_to_save) % 100 == 0:
            with open("data_sft.json", "w") as f:
                json.dump(data_to_save, f, indent=4)

    with open("data_sft.json", "w") as f:
        json.dump(data_to_save, f, indent=4)


if __name__ == "__main__":
    create_sft_data("./data.json", filter_label="TRAIN")
