import requests
import ast
import json
import numpy as np
from transformers import AutoTokenizer,AutoModelForCausalLM
from openai import OpenAI
from openai import AzureOpenAI
import os

def bulid_input(prompt, history=[]):
    system_format='<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>'
    user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>'
    assistant_format='<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>\n'
    history.append({'role':'user','content':prompt})
    prompt_str = ''
    # 拼接历史对话
    for item in history:
        if item['role']=='user':
            prompt_str+=user_format.format(content=item['content'])
        else:
            prompt_str+=assistant_format.format(content=item['content'])
    return prompt_str

def complete_text_llama(llm_model_pt,
                   tokenizer,
                   message, 
                   model=None,
                   max_tokens=2048, 
                   temperature=1, 
                   max_retry=1,
                   sleep_time=60,
                   json_object=False):

    # os.environ["AZURE_OPENAI_API_KEY"] = "419887436f6a4ba3af53360e309d7a73"

    # client = AzureOpenAI(api_key=os.getenv("OPENAI_API_KEY"), azure_endpoint="https://yyxzhj.openai.azure.com/", api_version="2024-02-01")

    # completion = client.chat.completions.create(
    #     model="Rising-test",
    #     messages=[{"role":"user","content":message}],
    #     temperature=0.1,
    #     top_p=1,
    #     max_tokens=1024
    # )
    
    # content = completion.choices[0].message.content  
    
    # print("=====================")        
    # print(content)
    # print("=====================")
    # return content
    
    # api_url = "http://0.0.0.0:8000/api/"

    # input_data = {
    #     "data": message
    # }

    # # 将输入数据转换为JSON格式的字符串
    # input_json = json.dumps(input_data)

    # # 发送POST请求
    # response = requests.post(api_url, data=input_json).json()['output']
    # output_vector = response

    messages = [
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content":message}
    	]

    # 调用模型进行对话生成
    history = []
    input_str = bulid_input(prompt=messages, history=history)
    inputs = tokenizer.encode(input_str, add_special_tokens=False, return_tensors='pt').cuda()

    generated_ids = llm_model_pt.generate(
        input_ids=inputs, max_new_tokens=512, max_length=max_tokens,
        do_sample=True, 
        top_p=0.9, temperature=0.1, repetition_penalty=1.1, eos_token_id=tokenizer.encode('<|eot_id|>')[0]
    )
    result = generated_ids.tolist()[0][len(inputs[0]):]
    inputs = inputs[0]

    response = tokenizer.decode(result)

    generated_text = response.strip().replace('<|eot_id|>', "").replace('<|start_header_id|>assistant<|end_header_id|>\n\n', '').strip()

    print(generated_text)

    return generated_text

def complete_text_qwen(llm_model_pt,
                   tokenizer,
                   message, 
                   model=None,
                   max_tokens=2048, 
                   temperature=1, 
                   max_retry=1,
                   sleep_time=60,
                   json_object=False):
    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
    ]

    # 调用模型进行对话生成
    input_ids = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors="pt").to('cuda')
    generated_ids = llm_model_pt.generate(model_inputs.input_ids,max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response