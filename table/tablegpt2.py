from transformers import AutoModelForCausalLM, AutoTokenizer

# Using pandas to read some structured data
import pandas as pd
from io import StringIO

# single table
EXAMPLE_CSV_CONTENT = """
"Loss","Date","Score","Opponent","Record","Attendance"
"Hampton (14–12)","September 25","8–7","Padres","67–84","31,193"
"Speier (5–3)","September 26","3–1","Padres","67–85","30,711"
"Elarton (4–9)","September 22","3–1","@ Expos","65–83","9,707"
"Lundquist (0–1)","September 24","15–11","Padres","67–83","30,774"
"Hampton (13–11)","September 6","9–5","Dodgers","61–78","31,407"
"""

csv_file = StringIO(EXAMPLE_CSV_CONTENT)
df = pd.read_csv(csv_file)

model_name = "/Users/baihailong/Documents/modelscope/LLM-Research/tablegpt_2_7B"

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

example_prompt_template = """Given access to several pandas dataframes, write the Python code to answer the user's question.

/*
"{var_name}.head(5).to_string(index=False)" as follows:
{df_info}
*/

Question: {user_question}
"""
question = "哪些比赛的战绩达到了40胜40负？"

prompt = example_prompt_template.format(
    var_name="df",
    df_info=df.head(5).to_string(index=False),
    user_question=question,
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(**model_inputs, max_new_tokens=512)
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(messages)
print(response)
