import argparse
import copy
import json
import logging
import sys
import dashscope
import random
import time
import os
import re
from datetime import datetime
import openai
from sentence_transformers import SentenceTransformer, util
from utils import *

import numpy as np
# os.environ["OPENAI_API_BASE"] = 'm.gptapi.us'
#"https://api.gptapi.us/v1"
#'https://www.gptapi.us'
# os.environ["OPENAI_API_KEY"] = "sk-VLeMdB2cyr0p6BIO8aEa2fA1466e46739fF03d4b320939Fe"
os.environ["OPENAI_API_KEY"] = "sk-7lDeR2hYGD3VwH2P509c67Bb0093478eA790641c19073964"
os.environ["OPENAI_API_BASE"] = 'https://api.gptapi.us/v1'

def decoder_for_gpt3(model,input):
    # GPT-3 API allows each users execute the API within 60 times in a minute ...
    time.sleep(3)

    #switch_key()
    model = model
    #model = "gpt-3.5-turbo"
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )
    response = None
    try:
        response = request_chat(client,model, input)
    except openai.APIStatusError as e:
        if e.status_code == 407:
            # 休眠2.5秒
            time.sleep(5)
            response = request_chat(client,model, input)
        else:
            print(e)
    except AttributeError as exception:
        time.sleep(5)
        response = request_chat(client, model, input)
        # print(exception)
        # print(type(exception))
        
    if response != None:
        print("====================")
        print(response)
        print("====================")
        return response.choices[0].message.content
    else:
        return response

def request_chat(client,model,input):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": input}
        ],
        max_tokens=1500,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response
             
def get_qwen_response(
        model: str,
        prompt: str,
        max_length: int
):
    # system:
    # "You should use your knowledge to answer the question to the best of your ability, not refuse to answer, even though you think you don't have the knowledge about the question. "
    # "Try to answer the question rather than requesting further information. "
    # "Refer to the example provided below to answer the question. "
    message = [
            {
                "role": "system",
                "content": "Refer to the examples provided below to answer the question.Do not refuse to answer.Do not request further information."
             },
            {"role": "user", 
             "content": f"{prompt}"
             }
        ]
    
    try:
            response = dashscope.Generation.call(
                model= model,
                messages=message,
                result_format='message',
                max_tokens  = max_length,

                #api_key = "sk-dcdfae8650244eb0ab76e7d3f00d9d18",
                ##jj
                #api_key = "sk-0f4b10ac2ec04411a5d8740ec6f4467b",
                #zkw：
                api_key = "sk-4cecd5e8603a458c8f106cca62e66092",
                #api_key = "sk-2a511609d85d4ca5aba830ca9e8350a3",
                #api_key = "sk-dd53979160dc43b0ad7bd1fccbb4f8f6",
                temperature=0.1,
                top_p=0.6
        )
            return response.output.choices[0]['message']['content']
    except:
            return ('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message))


def get_sentence_encoder():
    model = "all-MiniLM-L6-v2"
    return SentenceTransformer(model)

# 对文本进行标准化处理
def general_postprocess(text: str) -> str:
    # Cut off the first newline, period, or comma
    truncated_text = re.split(r'[\n.,]', text, 1)[0]

    # Remove punctuation
    no_punctuation = re.sub(r'[^\w\s]', '', truncated_text)

    # Remove article
    no_articles = re.sub(r'\b(a|an|the)\b',
                         '',
                         no_punctuation,
                         flags=re.IGNORECASE)

    # Remove duplicated blank spaces
    cleaned_text = re.sub(r'\s+', ' ', no_articles).strip()

    return cleaned_text

#对参考答案进行格式处理
def reference_format(task,answer):
    if task == "gsm8k":
        match = re.search(r'\d+',answer[::-1])  # 从字符串末尾开始匹配数字
        if match:
            extracted_number = match.group()[::-1]  # 将结果反转得到正确的数字
    return extracted_number


def create_question_cluster_demons(dataset_path:str,question_embedding,demo_num:int):

    #读取该任务的语料库
    with open(dataset_path, 'r',encoding="utf-8") as f:
        demo_datas = json.load(f)
    
    question_embedding = np.array(question_embedding, dtype=float)

    # 定义一个存放测试问题与所有demo问题的相似度的列表
    similarity_list = []
    for cluster_idx in range(len(demo_datas)):
        #print(demo_datas[cluster_idx])
        train_samples = demo_datas[cluster_idx]["train_samples"]
        inner_similarity_list = []
        for sample_idx in range(len(train_samples)):
            # 计算当前测试问题的embedding与遍历到的demo问题的embedding的相似度
            demo_embedding = np.array(train_samples[sample_idx]["embedding"], dtype=float)
            similarity = util.cos_sim(question_embedding, demo_embedding)[0, 0]
            inner_similarity_list.append(similarity)
        similarity_list.append(inner_similarity_list)
    # 找到最长的相似度列表的长度
    max_len = max(len(similarity_list[i]) for i in range(len(similarity_list)))
    # 对齐长度并填充0
    for i in range(len(similarity_list)):
        similarity_list[i] += [0] * (max_len - len(similarity_list[i]))
    # 将相似度列表转换成numpy数组
    similarity_array = np.array(similarity_list)
    # 找到这个二维列表中最相似的demo_num个问题的索引，也就是找到这个二维列表中值最大的demo_num个值的索引
    x = 1 + demo_num
    index = np.argsort(similarity_array.ravel())[:-x:-1]
    positions = np.unravel_index(index, similarity_array.shape)
    positions_2d = np.column_stack(positions)
    question_demonstrations = []
    sample_pre = {}
    for i in range(len(positions_2d)):
            indexs = positions_2d[i]
            cluster_id = indexs[0]
            sample_id = indexs[1]
            sample = demo_datas[cluster_id]["train_samples"][sample_id]
            if similarity_array[cluster_id, sample_id] != 0:
                sample_pre = {
                "question" : sample["question"],
                "elements" : sample["elements"],
                "questions" : sample["questions"],
                "cot" : sample["cot"],
                "answer" : sample["gold_ans"]
            }
            question_demonstrations.append(sample_pre)
    
    return question_demonstrations

def demo_format(question_demonstrations,task):
    demos = ""
    for sample in question_demonstrations:
        questions = sample['questions'].replace('\n', '')
        passage = sample['cot'].replace('\n', '')
        answer = sample['answer']
    
        if task in ('gsm8k'):
            demo =  f"Question:{sample['question']}\n" \
                    f"Answer:Let's think step by step.\n\n" \
                    f"Step 1:Key elements of the question:{sample['elements']}.\n" \
                    f"Step 2:Decompose the question into new questions:{questions}\n" \
                    f"Step 3:Analyzing these new questions can inferr that:{passage}\n\n" \
                    f"Therefore, the answer (decimal numerals) is {answer}\n\n" 
        
        elif task in ('aqua'):
            demo =  f"Question:{sample['question']}\n" \
                    f"Answer:Let's think step by step:\n\n" \
                    f"Step 1:Key elements of the question:{sample['elements']}.\n" \
                    f"Step 2:Decompose the question into new questions:{questions}\n" \
                    f"Step 3:Analyzing these new questions can inferr that:{passage}\n\n" \
                    f"Therefore, among A through E, the answer is {answer}\n\n"     
            
        elif task in ('hotpot','musique'):
            demo =  f"Question:{sample['question']}\n" \
                    f"Answer:Let's think step by step:\n\n" \
                    f"Step 1:Key elements of the question:{sample['elements']}.\n" \
                    f"Step 2:Decompose the question into new questions:{questions}\n" \
                    f"Step 3:Analyzing these new questions can inferr that:{passage}\n\n" \
                    f"Therefore, the answer is {answer}\n\n"  
            
        demos += demo
    
    return demos

def answer_cleansing_new(task,pred,method):
    
    #目前仅针对qwen的输出风格进行的答案提取

    if task in ('gsm8k'):
        answer_trigger = "Therefore, the answer (decimal numerals) is"
    elif task in ('aqua'):
        answer_trigger = "Therefore, among A through E, the answer is"
    elif task in ('hotpot','musique'):
        answer_trigger = "Therefore, the answer is"

    pred = pred.strip().split('\n')[-1]
    
    if method in ("wh-cot","few_shot", "few_shot_cot"):
        preds = pred.split(answer_trigger)
        pred = preds[-1]
    
    if task in ("aqua", "commonsensqa"):
        pred = pred.split('answer to the question is')[-1]
        pred = pred.split('answer is')[-1]
        pred = re.findall(r'A|B|C|D|E', pred)

    elif task in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
        pred = pred.split('answer to the question is')[-1]
        pred = pred.split('answer is')[-1]
        pred = pred.split('Answer:')[-1] 
        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

    elif task in ('hotpot','musique'):
        pred = pred.split('the answer is:')[-1]
        pred = pred.split('answer to the question is')[-1]
        pred = pred.split('The answer would be')[-1]
        pred = pred.split('answer is')[-1]
        pred = pred.split('Answer:')[-1] 
        pred = pred.strip(':')
        pred = [pred]
    else:
        raise ValueError("dataset is not properly defined ...")

    if len(pred) == 0:
        pred = ""
    else:
        #
 
        pred = pred[0]

        # if method in ("few-shot", "few-shot-cot", "wh-cot","zero-shot"):
        #     if answer_flag:
        #         # choose the first element in list ...
        #         pred = pred[0]
        #     else:
        #         # choose the last element in list ...
        #         pred = pred[-1]
        # elif method in ("zero-shot-cot"):
        #     # choose the first element in list ...
        #     pred = pred[0]
        # else:
        #     raise ValueError("method is not properly defined ...")
        
    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]

    return pred

def main():
    
    args = parse_arguments()
    print('*****************************')
    print( datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(args)
    print('*****************************')
    
    #获取模型、任务和数据集大小
    model = args.model
    task = args.task
    test_data_size = args.test_data_size
    demo_num = args.demo_num
    method = args.method

    #记录日志
    current_time = datetime.now().strftime("%m%d-%H%M%S")

    log_file_path = f"./eval-log/{model}/{task}/"
    if not os.path.exists(log_file_path):
            os.makedirs(log_file_path)  
    log_file_name = f"{current_time}-{task}-{method}.log"
    log_file = os.path.join(log_file_path, log_file_name)

    # 重定向标准输出到日志文件
    sys.stdout = open(log_file, "w", encoding='utf-8')

    args = parse_arguments()
    print('*****************************')
    print( datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(args)
    print('*****************************')
    
    #初始化
    encoder = get_sentence_encoder()
    demo_dataset_path = f"./demos/{task}/demo_embedding.json"
    correct_list = []
    num = 0
    all_em = 0
    all_f1 = 0
    em_score = 0
    f1_score = 0
    if model in ("qwen-72b-chat"):
        max_length = 2000
    elif model in ("qwen-14b-chat", "qwen-7b-chat", "qwen-turbo"):
        max_length = 1500

    if args.method == "few-shot":
        demo = create_demo_text(task, cot_flag=False)
    elif args.method == "few-shot-cot":
        demo = create_demo_text(task, cot_flag=True)
    else:
        pass

    # 读取test数据集
    # 测试gpt
    data = []

    if task == "gsm8k":
        file_path = './dataset/test-s/gsm8k_test_160.json'
        direct_answer_trigger_for_zeroshot = "The answer (decimal numerals) is"



    elif task == "aqua":
        file_path = './dataset/test-s/aqua_test.json'
        direct_answer_trigger_for_zeroshot = "Among A through E, the answer is"
    elif task == "hotpot":
        file_path = './dataset/test-s/hotpot_test_120.json'
        direct_answer_trigger_for_zeroshot = "The answer is"
    elif task == "musique":
        file_path = './dataset/test-s/musique_test_120.json'
        direct_answer_trigger_for_zeroshot = "The answer is"
    else:
        raise ValueError("task is not properly defined ...")
    
    with open(file_path, "r", encoding='utf-8') as file:
        data = json.load(file)
    
    #infer
    for i in range(len(data)):
        samples = data[i]['test_samples']
        for j in range(len(samples)):
            sample = samples[j]
            question = sample['question']
            
            print(f"*****************************")
            if method == "zero-shot":
                input = f"Question:{question}\n" \
                        f"Answer: {direct_answer_trigger_for_zeroshot}"
                #max_length = 50
            
            elif method == "zero-shot-cot":
                input = f"Question:{question}\n" \
                        f"Answer: Let's think step by step." 
            
            elif method in ("few-shot", "few-shot-cot"):
                input = f"{demo}"\
                        f"Question:{question}\n" \
                        f"Answer:" 

            elif method == "wh-cot":
                #计算问题的Embedding
                question_embedding = encoder.encode(question)
                #返回与问题最相似的demo组,含num个样本
                question_demonstrations= []
                question_demonstrations=create_question_cluster_demons(demo_dataset_path,question_embedding,demo_num)
                
                #单个demo样式
                # demo = {
                # "question" : sample["question"],
                # "elements" : sample["elements"],
                # "questions" : sample["questions"],
                # "cot" : sample["cot"],
                # "answer" : sample["gold_ans"]
                # }

                format_demo = demo_format(question_demonstrations,task)
                input = f"{format_demo}"\
                        f"Question:{question}\n" \
                        f"Answer:Let's think step by step.\n" 
            else:
                raise ValueError("method is not properly defined ...")    

            #output = get_qwen_response(model, input, max_length)
            output = decoder_for_gpt3(model=model, input=input)

            if method == "zero-shot-cot":
                input = f"{input}\n{output}\nTherefore, the answer is"
                output = decoder_for_gpt3(model=model, input=input)
        
            print(f"Input:\n{input}")
            print(f"Output:\n{output}")

            #提取答案
            pred = answer_cleansing_new(task, output, method)
            pred = extract_answer(pred)

            # Checking answer ...
            if task in ("hotpot", "musique"):
                f1_score = compute_f1(pred,sample['gold_ans'])
                em_score = compute_em(pred,sample['gold_ans'])
                print(f"EM:{em_score} , F1:{f1_score}")
                if em_score == 1.0:
                    correct = 1
                else:
                    correct = 0
                
            elif task in ("aqua","gsm8k"):
                if pred == "":
                    pred = 0
                pred = float(pred)
                sample['gold_ans'] = float(sample['gold_ans'].replace(",", ""))
                correct = (np.array([pred]) == np.array([sample['gold_ans']])).sum().item()
     
            print(f"Output_clean:{pred}")
            print(f"answer:{sample['gold_ans']}")

            correct_list.append(correct)
            all_em += em_score
            all_f1 += f1_score
            num += 1
            
            print(f"correct:{correct}")
            print(f"已完成：{num}")
            print("总正确数 : {}".format(sum(correct_list)))
            print("*****************************") 
            sys.stdout.flush()
            
    # Calculate accuracy ...
    accuracy = (sum(correct_list) * 1.0 / num) * 100
    print("EM: {:.2f}, F1: {:.2f}".format(all_em * 100 / num, all_f1 * 100 / num))
    print("correct_num : {}".format(sum(correct_list)))
    print("total_num : {}".format(num))
    print("accuracy : {}".format(accuracy))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Creat Dataset For Symbolic Task Reasoning")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",choices=["gpt-3.5-turbo", "qwen-7b-chat","qwen-14b-chat",'qwen-72b-chat','qwen-turbo'], help="select model")
    #parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument("--task", type=str, default="musique", choices=["musique","hotpot","aqua","triviaqa","multiarith","gsm8k","strategyqa","coin_flip", "last_letters"], help="")
    parser.add_argument("--test_data_size", type=int, default=120, help="the number of test data")
    parser.add_argument("--demo_num", type=int, default=2, help="the demo number of each question")
    parser.add_argument("--method", type=str, default="zero_shot", choices=["few-shot", "few-shot-cot","zero-shot","wh-cot","zero-shot-cot"], help="the method")
    parser.add_argument("--temperature", type=float, default=0.1, help="model temperature")
    parser.add_argument("--top-p", type=float, default=0.6, help="model top-p")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()