import argparse
import json
import logging
import sys
import dashscope
import random
import time
import os
import re
from datetime import datetime
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from utils import *

def get_sentence_encoder():
        model = "all-MiniLM-L6-v2"
        return SentenceTransformer(model)
def add_embedding():
    #为数据集添加Embedding
    encoder = get_sentence_encoder()
    dataset_path = "E:\LLM\zero-shot-cot-cmk\demo/multiarith/202403241514\demo_dataset.jsonl"
    demo =[]
    with open(dataset_path, 'r',encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                demo.append(sample)
    for i in range(len(demo)):
        question =  demo[i]['question']
        embedding = encoder.encode(question)
        demo[i]["embedding"] = embedding.tolist()


    dataset_embedding_path = "E:\LLM\zero-shot-cot-cmk\demo/multiarith/202403241514\demo_embedding.json"
    with open(dataset_embedding_path, 'w',encoding="utf-8") as f:
        json.dump(demo,f,indent=4)

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


def get_qwen_response(
        model: str,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.85,
        # top_p: float = 1,
):
    message = [
            {
                "role": "system",
                "content": "You should use your knowledge to answer the question to the best of your ability.Do not refuse to answer.Do not requesting further information. "
             },
            {   "role": "user", 
                "content": prompt
            }
        ]
    
    try:
            response = dashscope.Generation.call(
                model= model,
                messages=message,
                result_format='message',
                #api_key = "sk-dcdfae8650244eb0ab76e7d3f00d9d18"

                #kw
                api_key = "sk-dd53979160dc43b0ad7bd1fccbb4f8f6",
                #max_tokens=max_tokens,
                #temperature=temperature,
                #stop=stop,
                #top_p=top_p
        )
            # print("===================")
            # print(response.output.choices[0]['message']['content'])
            # print("===================")
            return response.output.choices[0]['message']['content']
    except:
            return ('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message))


def gen_demo_cot(model,task,sample):
    #print(sample)
    model = 'qwen-72b-chat'
    answer_list = sample['answer']
    question = sample['question']
    #print(answer_list)

    #提取问题的元素
    #元素提取的example有待优化
    examples = "Question1:Who was the next British Prime Minister after Arthur Balfour?\nElements:British Prime Minister, Arthur Balfour, the next British Prime Minister.\nQuestion2:A robe takes 2 bolts of blue fiber and half that much white fiber.How many bolts in total does it take?\nElements:A robe, 2 bolts of blue fiber, half that much white fiber, bolts.\nQuestion3:A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?\nElements:A revolving door, direction travel, security measure"
    element_prompt = f"Identify key elements from the following questions,Please refer to the example I provided.\nExample:\n{examples}\n\nIdentify key elements from the following questions,Only give me the elements and do not output any other words.\n\nQuestion:{sample['question']}\nElements:"
    #print(prompt)
    elements= get_qwen_response(model,element_prompt)
    # print(f"Question:{sample['question']}")
    print(f'Elements:{elements}')
    print("==========================") 
    
    #预测若与参看答案不一致，则重新生成问题、段落、预测
    count= 0
    while True: 
        #根据元素生成问题
        #生成问题数,默认为6;问题方向目前为6W1H，共7个
        #num_question = 'six'
        num_question = '8'
        #问题方向目前为6W1H，共7个
        if task in ('triviaqa'):
            num_question = '6'
            direction = "-Which: Which element and direction to choose.\n-Why: Explore the purpose or reason behind something.\n-What: Refers to the subject or topic being discussed.\n-Who: Identify the people or entities involved.\n-Where: Specify the location or place.\n-When: Indicating the time or timeframe.\n-How: Describe the method or process."
        
        #数学推理问题：方向目前为6W2H，共8个
        elif task in ('multiarith','gsm8k'):
            num_question = '8'
            direction = "-Which: Identify the specific mathematical operations or concepts involved in the problem, such as addition, subtraction, multiplication, division, algebraic manipulation, etc.\n-What: Define the mathematical operations or procedures involved in solving the problem.\n-Who: May involve individuals or entities affected by the measurement.\n-Where: Specify the context or setting in which the mathematical problem arises.\n-When: Establish the timeframe or temporal context within which the problem is relevant or needs to be solved.\n-Why: Understand the purpose or objective behind solving the mathematical problem.\n-How: Break down the step-by-step process or strategies for solving the mathematical problem efficiently and accurately.\n-How much: Examine the numerical value or extent of something, determine any quantitative aspects of the problem, such as the number of steps involved, the amount of time required, or the quantity of data to be processed."
        
        
        gen_question_prompt = f"Decompose the original question into {num_question} new questions from the given direction, each of question needs to include at least one extracted element.\nOriginal question:\n{question}\nElements:\n{elements}\nDirection:\n{direction}\n\nOnly give me these new questions and do not output others.One question per line."
        #print(gen_question_prompt)
        questions=get_qwen_response(model,gen_question_prompt)
        print(f"questions:\n{questions}")
        print("==========================")
        
        #生成答案段落
        if task == "gsm8k":
            gen_passage_prompt = f"Answer these questions and form a passage with the answers.The passage should include some calculation steps or formulas to solve these questions.\nQuestion:{questions}\n\nOnly give me the passage and do not output others.The passage should include some calculation steps or formulas to solve these questions."
        else:
            gen_passage_prompt = f"Answer these questions and form a passage with the answers.\nQuestion:{questions}\n\nOnly give me the passage and do not output othersLet's think step by step.\nPassage:"
        
        
        
        passage = get_qwen_response(model,gen_passage_prompt)
        #print(gen_passage_prompt)
        print(f"passage:\n{passage}")
        print("==========================")
        
        #根据生成的段落给出prediction
        if task == 'triviaqa':
            gen_prediction_prompt = f"Answer the question based on the given passages and you knowledge . Only give me the answer and do not output any other words.The answer should be as simple as possible.\n\nThe following are given passages.\n{passage}\n\nAnswer the question based on the given passages and you knowledge.Only give me the answer and do not output any other words.The answer should be as simple as possible.\n\nQuestion: {sample['question']}\nAnswer:"
        
        
        elif task in ('multiarith','gsm8k'):
            gen_prediction_prompt = f"Answer the question based on the given passages and you knowledge . Only give me the numerical answer and do not output any other words.\n\nThe following are given passages.\n{passage}\n\nAnswer the question based on the given passages and you knowledge.Only give me the numerical answer and do not output any other words.\n\nQuestion: {sample['question']}\nAnswer:Let's think step by step."
        
        prediction = get_qwen_response(model,gen_prediction_prompt)
        # print(f"prediction:\n{prediction}")
        # print("==========================")
        
        
        #对prediction进行处理，并判断prediction是否和参考答案一致
        same = False
        pre = prediction
        if task == 'triviaqa':
            pre = pre.lower()
            pre = pre.strip()
            pre = general_postprocess(pre)
            answer_list = [general_postprocess(answer).lower() for answer in answer_list]
            print(f"pre:{pre}")
            print(answer_list)
            if any(pre in answer for answer in answer_list):
                same = True
                # print("常识预测正确,保存cot")
                # print("==========================")  
                break
            else:
             print("预测不正确，重新生成")
             print(f"question:{question}")
             print(f"prediction:{prediction}")
             print(f"pre:{pre}")
             print(f"answer_list:{answer_list}")
             print("==========================")  
             count += 1
        
        elif task == 'multiarith':
            #结果处理
            matches = re.findall(r'\b\d+(?:\.\d+)?\b', pre)
            # 如果找到匹配项，则选择最后一个数字
            if matches:
                pre = matches[-1]
            pre = float(pre)
            answer_list = float(answer_list)

            if pre == answer_list :
                same = True
                # print("数学推理预测正确,保存cot")
                # print("==========================")  
                break
            else:
             print("数学预测不正确，重新生成")
             print(f"question:{question}")
             print(f"prediction:{prediction}")
             print(f"pre:{pre}")
             print(f"answer_list:{answer_list}")
             print("==========================")  
             count += 1
        
        elif task == 'gsm8k':
            #结果处理
            pre_matches = re.findall(r'\b\d+(?:\.\d+)?\b', pre)
            # 如果找到匹配项，则选择最后一个数字
            if pre_matches:
                pre =pre_matches[-1]
            pre = float(pre)
            
            # 对参考答案使用正则表达式提取最后的数字部分
            match = re.search(r'\d+',answer_list[::-1])  # 从字符串末尾开始匹配数字
            if match:
                extracted_number = match.group()[::-1]  # 将结果反转得到正确的数字
            answer_number = float(extracted_number)

            # print(f"pre:{pre}")
            # print(f"answer_number:{answer_number}")
            if pre == answer_number :
                same = True
                # print("数学推理预测正确,保存cot")
                # print("==========================")  
                break
            else:
             print("数学预测不正确，重新生成")
             print(f"question:{question}")
             print(f"prediction:{prediction}")
             print(f"pre:{pre}")
             print(f"answer_list:\n{answer_list}")
             print("==========================")  
             count += 1
        
        #重复次数控制
        if count == 3:
             print(f"重复生成{count}次后保存")
             print("==========================")  
             break
    sample['prediction'] = prediction
    sample['same'] = same              
    sample['elements'] = elements
    sample['questions'] = questions
    sample['cot'] = passage
    
    
    #print("==========================")
    #print(sample)

    return sample



def gen_cluster_cot(model,task,sample):
    answer_list = sample['gold_ans']
    question = sample['question']
    if task == 'aqua':
        question_1 = question.split('\n Answer Choices:')[0]

    #提取问题的元素
    #元素提取的example有待优化
    examples = "Question1:Who was the next British Prime Minister after Arthur Balfour?\nElements:British Prime Minister, Arthur Balfour, the next British Prime Minister.\nQuestion2:A robe takes 2 bolts of blue fiber and half that much white fiber.How many bolts in total does it take?\nElements:A robe, 2 bolts of blue fiber, half that much white fiber, bolts.\nQuestion3:A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?\nElements:A revolving door, direction travel, security measure"
    element_prompt = f"Identify key elements from the following questions,Please refer to the example I provided.\nExample:\n{examples}\n\nIdentify key elements from the following questions,Only give me the elements and do not output any other words.\n\nQuestion:{question}\nElements:"
    elements= get_qwen_response(model,element_prompt)
    # print(f"Question:{sample['question']}")
    # print(f'Elements:{elements}')
    # print("==========================") 
    
    #生成问题、段落、预测
    #预测若与参看答案不一致，则重新生成
    count= 0
    while True: 
        #根据元素生成问题
        #问题方向目前为6W1H，共7个
        if task in ('triviaqa','strategyqa','hotpot','musique',"2wikim"):
            num_question = '6'
            direction = "-Which: Which element and direction to choose.\n-Why: Explore the purpose or reason behind something.\n-What: Refers to the subject or topic being discussed.\n-Who: Identify the people or entities involved.\n-Where: Specify the location or place.\n-When: Indicating the time or timeframe.\n-How: Describe the method or process."
        
        #数学推理问题：方向目前为6W2H，共8个
        elif task in ('multiarith','gsm8k','aqua', "svamp"):
            num_question = '6'
            direction = "-Which: Identify the specific mathematical operations or concepts involved in the problem, such as addition, subtraction, multiplication, division, algebraic manipulation, etc.\n-What: Define the mathematical operations or procedures involved in solving the problem.\n-Who: May involve individuals or entities affected by the measurement.\n-Where: Specify the context or setting in which the mathematical problem arises.\n-When: Establish the timeframe or temporal context within which the problem is relevant or needs to be solved.\n-Why: Understand the purpose or objective behind solving the mathematical problem.\n-How: Break down the step-by-step process or strategies for solving the mathematical problem efficiently and accurately.\n-How much: Examine the numerical value or extent of something, determine any quantitative aspects of the problem, such as the number of steps involved, the amount of time required, or the quantity of data to be processed."
        
        gen_question_prompt = f"Decompose the original question into {num_question} new questions from the given direction, each of question needs to include at least one extracted element.\nOriginal question:\n{question}\nElements:\n{elements}\nDirection:\n{direction}\n\nOnly give me these new questions and do not output others."
        #print(gen_question_prompt)
        questions=get_qwen_response(model,gen_question_prompt)
        # print(f"questions:\n{questions}")
        # print("==========================")
        
        #生成答案段落
        if task in ("gsm8k","aqua", "svamp"):
            gen_passage_prompt = f"Answer these questions and form a passage with the answers.The passage should include some calculation steps or formulas to solve these questions.\nQuestion:{questions}\n\nOnly give me the passage and do not output others.The passage should include some calculation steps or formulas to solve these questions."
        else:
            gen_passage_prompt = f"Answer these questions and form a passage with the answers.\nQuestion:{questions}\n\nOnly give me the passage and do not output others."
        
        passage = get_qwen_response(model,gen_passage_prompt)
        #print(gen_passage_prompt)
        # print(f"passage:\n{passage}")
        # print("==========================")
        
        #根据生成的段落给出prediction
        if task in ('triviaqa','hotpot','musique',"2wikim"):
            gen_prediction_prompt = f"Answer the question based on the given passages and you knowledge . Only give me the answer and do not output any other words.The answer should be as simple as possible.\nThe following are given passages.\n{passage}\n\nAnswer the question based on the given passages and you knowledge.Only give me the answer and do not output any other words.The answer should be as simple as possible.\n\nQuestion: {sample['question']}\nThe answer is:"
        
        elif task in ('multiarith','gsm8k', "svamp"):
            gen_prediction_prompt = f"Answer the question based on the given passages and you knowledge . Only give me the numerical answer and do not output any other words.\nThe following are given passages:\n{passage}\n\nAnswer the question based on the given passages and you knowledge.Only give me the numerical answer and do not output any other words.\n\nQuestion: {sample['question']}\nAnswer:The answer is"
        
        elif task in ('strategyqa'):
            gen_prediction_prompt = f"Answer the question based on the given passages and you knowledge.Only answer \"yes\",or \"no\". Do not provide any explanation.\nThe following are given passages:\n{passage}\n\nAnswer the question based on the given passages and you knowledge.Only answer \"yes\",or \"no\". Do not provide any explanation.\n\nQuestion: {sample['question']}\nAnswer:(Yes or No)"

        elif task in ("aqua"):
            gen_prediction_prompt = f"The following are given passages.\n{passage}\n\nChoose the correct answer of the question from the Answer Choices.\n\nQuestion: {sample['question']}\nThe output format should only contain letters such as A, B, C, D or E.Therefore, among A through E, the answer is"

        prediction = get_qwen_response(model,gen_prediction_prompt)
        # print(f"prediction:\n{prediction}")
        # print("==========================")
        
        
        #对prediction进行处理，并判断prediction是否和参考答案一致
        same = False
        pre = prediction
        if task == 'triviaqa':
            pre = pre.lower()
            pre = pre.strip()
            pre = general_postprocess(pre)
            answer_list = [general_postprocess(answer).lower() for answer in answer_list]
           # print(f"pre:{pre}")
            #print(answer_list)
            if any(pre in answer for answer in answer_list):
                same = True
                # print("常识预测正确,保存cot")
                # print("==========================")  
                break
            else:
             print("预测不正确，重新生成")
             print(f"question:{question}")
             print(f"prediction:{prediction}")
             print(f"pre:{pre}")
             print(f"answer_list:{answer_list}")
             print("==========================")  
             count += 1
        
        elif task == 'multiarith':
            #结果处理
            matches = re.findall(r'\b\d+(?:\.\d+)?\b', pre)
            # 如果找到匹配项，则选择最后一个数字
            if matches:
                pre = matches[-1]
            pre = float(pre)
            answer_list = float(answer_list)

            if pre == answer_list :
                same = True
                # print("数学推理预测正确,保存cot")
                # print("==========================")  
                break
            else:
             print("数学预测不正确，重新生成")
             print(f"question:{question}")
             print(f"prediction:{prediction}")
             print(f"pre:{pre}")
             print(f"answer_list:{answer_list}")
             print("==========================")  
             count += 1
        
        elif task in ('gsm8k', "svamp"):
            #结果处理
            pre = pre.split('answer to the question is')[-1]
            pre = pre.split('answer is')[-1]
            pre = pre.split('Answer:')[-1] 
            pre = pre.replace(",", "")
            pre = [s for s in re.findall(r'-?\d+\.?\d*', pre)]
            
            #pre_matches = re.findall(r'\b\d+(?:\.\d+)?\b', pre)
            # 如果找到匹配项，则选择最后一个数字
            # if pre_matches:
            #     pre =pre_matches[-1]

            if len(pre) == 0:
                pre = 0
            else:
                pre = pre[0]
            pre = float(pre)
            
            # 对参考答案使用正则表达式提取最后的数字部分
            # match = re.search(r'-?\d+',answer_list[::-1])  # 从字符串末尾开始匹配数字
            # if match:
            #     extracted_number = match.group()[::-1]  # 将结果反转得到正确的数字
            answer_number = float(answer_list.replace(",", ""))

            # print(f"pre:{pre}")
            # print(f"answer_number:{answer_number}")
            print(f"question:{question}")
            #print(f"input:\n{gen_prediction_prompt}")
            print(f"prediction:{prediction}")
            print(f"pre:{pre}")
            print(f"answer:{answer_number}")
            if pre == answer_number :
                same = True
                # print("数学推理预测正确,保存cot")
                # print("==========================")  
                break
            else:
             print("数学预测不正确，重新生成")
             print("==========================")  
             count += 1
            
        elif task == 'strategyqa':
            pre = pre.lower()
            pre = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", pre)
            pre = pre.split(" ")
            pre = [i for i in pre if i in ("yes", "no")]
            if len(pre) == 0:
                pre = ""
            if pre != "":
                if pre[-1] == ".":
                    pre = pre[:-1]   
            correct = (np.array([pre]) == np.array([answer_list])).sum().item()
            if correct == 1:
                same = True
                # print("数学推理预测正确,保存cot")
                # print("==========================")  
                break
            else:
             print("预测不正确，重新生成")
             print(f"question:{question}")
             print(f"prediction:{prediction}")
             print(f"pre:{pre}")
             print(f"answer_list:{answer_list}")
             print("==========================")  
             count += 1
        
        elif task == 'aqua':
            pre = pre.strip().split('\n')[-1]
            pre = pre.split('Therefore,')[-1]
            pre = pre.split('answer to the question is')[-1]
            pre = pre.split('answer is')[-1]
            pre = re.findall(r'A|B|C|D|E', pre)
            if len(pre) == 0:
                pre = ""
            if pre != "":
                if pre[-1] == ".":
                    pre = pre[:-1]   
            correct = (np.array([pre]) == np.array([answer_list])).sum().item()
            if correct == 1:
                same = True
                # print("数学推理预测正确,保存cot")
                # print("==========================")  
                break
            else:
             print("预测不正确，重新生成")
             print(f"question:{question}")
             print(f"prediction:{prediction}")
             print(f"pre:{pre}")
             print(f"answer_list:{answer_list}")
             print("==========================")  
             count += 1
             sys.stdout.flush()


        elif task in ('hotpot','musique',"2wikim"):
            pre = pre.strip().split('\n')[-1]
            pre = pre.split('The answer is:')[-1]
            pre = pre.split('answer to the question is')[-1]
            pre = pre.split('answer is')[-1]
            pre = pre.split('The answer would be')[-1]
            pre = pre.split('Answer:')[-1] 
            pre = pre.strip(':')
            pre = extract_answer(pre)
            
            f1 = compute_f1(pre,answer_list)
            
            if f1 >= 0.4:
                same = True
                break
            else:
             print("预测不正确，重新生成")
             print(f"f1:{f1}")
             print(f"question:{question}")
             print(f"prediction:{prediction}")
             print(f"pre:{pre}")
             print(f"answer_list:{answer_list}")
             print("==========================")  
             count += 1
             sys.stdout.flush()

        #重复次数控制
        if count == 3:
             print(f"重复生成{count}次后保存")
             print("==========================")  
             break
        sys.stdout.flush()
    sample['elements'] = elements
    sample['questions'] = questions
    sample['cot'] = passage
    sample['pred_ans'] = prediction
    sample['same'] = same              
    

    return sample


def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    #获取模型、任务和数据集大小
    model = args.model
    task = args.task
    num_samples = args.dataset_size

    #记录日志
    current_time = datetime.now().strftime("%m%d-%H%M%S")

    log_file_path = "./demo-log/gen-cot/"
    if not os.path.exists(log_file_path):
            os.makedirs(log_file_path)  
    log_file_name = f"{current_time}-{task}.log"
    log_file = os.path.join(log_file_path, log_file_name)

    # 重定向标准输出到日志文件
    sys.stdout = open(log_file, "w", encoding='utf-8')

    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')

    
    #创建保存目录及文件
    current_time = datetime.now().strftime("%Y%m%d%H%M")

    demo_dataset_path = f"./demos/{task}/{current_time}/"
    if not os.path.exists(demo_dataset_path):
        os.makedirs(demo_dataset_path)  
    save_file_path = os.path.join(demo_dataset_path, 'demo_dataset.json')    
    cache_file_path = os.path.join(demo_dataset_path, 'cache_demo_dataset.json')  
    # 读取数据集
    data = []
    num = 0
    if task == "triviaqa":
        file_path = './dataset/triviaqa/triviaqa-validation.jsonl'
        with open(file_path, "r", encoding='utf-8') as file:
            for _ in range(num_samples):
                line = file.readline().strip()
                if line:
                    sample = json.loads(line)
                    data.append(sample)
                
        #构造样本的cot过程
        for sample in data:
            demo = gen_demo_cot(model,task,sample)
            with open(save_file_path, 'a', encoding='utf-8') as f:
                 f.write(json.dumps(sample,indent=4) + '\n') 
            num += 1
            print(f"*****************************")
            print(f"生成样本:\n {demo}")
            print(f"已完成进度: {num}/{num_samples}")
            print("*****************************") 
            
    elif task == "multiarith":
        file_path = './dataset/MultiArith/train_sample.json'
        with open(file_path, "r", encoding='utf-8') as file:
            data = json.load(file)[:num_samples]
        
        #数据预处理
        pro_data = []
        for orgin_sample in data:
            sample = {
                "question": orgin_sample["sQuestion"],
                "answer": str(orgin_sample["lSolutions"][0])  # 注意：这里需要取列表中的第一个元素并转换为字符串
            }
            pro_data.append(sample)
        #构造样本的cot过程
        for sample in pro_data:
            demo = gen_demo_cot(model,task,sample)
            with open(save_file_path, 'a', encoding='utf-8') as f:
                 f.write(json.dumps(sample,indent=4) + '\n') 
            num += 1
            print(f"*****************************")
            # print(f"生成样本:\n {demo}")
            print(f"已完成进度: {num}/{num_samples}")
            print("*****************************") 
    
    elif task in ("gsm8k","svamp"):
        file_path = f'./dataset/train\{task}_train.json'
        with open(file_path, "r", encoding='utf-8') as file:
                    data = json.load(file)
        
        #构造样本的cot过程
        for i in range(len(data)):
            samples = data[i]['train_samples']
            for j in range (len(samples)):
                sample = samples[j]
                sample = gen_cluster_cot(model,task,sample)
                data[i]['train_samples'][j] = sample

                #缓存文件，避免程序崩
                with open(cache_file_path, 'a', encoding='utf-8') as f:
                    json.dump(sample, f, indent=4)
                    f.write("\n")
                num += 1
                print(f"*****************************")
                print(f"生成样本:\n {sample}")
                print(f"已完成进度: {num}/{num_samples}")
                print("*****************************") 
                sys.stdout.flush()
        with open(save_file_path, 'a', encoding='utf-8') as f:
             json.dump(data, f, indent=4)
     
    elif task == 'strategyqa':
        #读取train数据集
        file_path = './new-dataset/train/strategyqa_train_30.json'
        data = []
        with open(file_path, "r", encoding='utf-8') as file:
            data = json.load(file)
        
        #构造样本的cot过程
        for i in range(len(data)):
            samples = data[i]['train_samples']
            for j in range(len(samples)):
                sample = samples[j]
                demo = gen_cluster_cot(model,task,sample)

                with open(save_file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(sample,indent=4) + '\n') 
                num += 1
                print(f"*****************************")
                print(f"生成样本:\n {demo}")
                print(f"已完成进度: {num}/300")
                print("*****************************") 
    
    elif task == 'aqua':
        #读取train数据集
        file_path = './new-dataset/train/aqua_train.json'
        data = []
        with open(file_path, "r", encoding='utf-8') as file:
            data = json.load(file)
        
        #构造样本的cot过程
        for i in range(len(data)):
            samples = data[i]['train_samples']
            for j in range(len(samples)):
                sample = samples[j]
                #print(f"sample----------:{sample}")
                sample = gen_cluster_cot(model,task,sample)
                # print(f"demo==========:{demo}")
                #print(f"sample==========:{sample}")
                data[i]['train_samples'][j] = sample

                #缓存文件，避免程序崩
                with open(cache_file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(sample,indent=4) + '\n') 

                num += 1
                print(f"*****************************")
                print(f"生成样本:\n {sample}")
                print(f"已完成进度: {num}/126")
                print("*****************************") 
                sys.stdout.flush()

        with open(save_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data,indent=4) + '\n') 
    
    elif task in ('hotpot','musique',"2wikim"):
        #读取train数据集
        file_path = f"./dataset/train/{task}_train.json"
        data = []
        with open(file_path, "r", encoding='utf-8') as file:
            data = json.load(file)

        # #cache文件
        # cache = []
        # with open("./demo\musique/202403291536\cache_demo_dataset.json", "r", encoding='utf-8') as file:
        #     cache = json.load(file)

        #构造样本的cot过程
        count= 0
        for i in range(len(data)):
            samples = data[i]['train_samples']
            for j in range(len(samples)):
                # count += 1
                # if count < 44:
                #     data[i]['train_samples'][j] = cache[count-1]
                # else:
                #     sample = samples[j]
                #     sample = gen_cluster_cot(model,task,sample)
                #     data[i]['train_samples'][j] = sample
                    sample = samples[j]
                    sample = gen_cluster_cot(model,task,sample)
                    data[i]['train_samples'][j] = sample

                    #缓存文件，避免程序崩
                    with open(cache_file_path, 'a', encoding='utf-8') as f:
                        json.dump(sample, f, indent=4)

                    num += 1
                    print(f"*****************************")
                    print(f"生成样本:\n {sample}")
                    print(f"已完成进度: {num}/120")
                    print("*****************************") 
                    sys.stdout.flush()

        with open(save_file_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data,indent=4) + '\n') 

def parse_arguments():
    parser = argparse.ArgumentParser(description="Creat Dataset For Symbolic Task Reasoning")
    parser.add_argument("--model", type=str, default='qwen-turbo',choices=['qwen-72b-chat','qwen-turbo'], help="select model")
    #parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument("--task", type=str, default="svamp", choices=["svamp","2wikim","musique","hotpot","aqua","triviaqa","multiarith","gsm8k","straregyqa","coin_flip", "last_letters"], help="")
    parser.add_argument("--dataset_size", type=int, default=120, help="")
    #parser.add_argument("--names_in_sample", type=int, default=4, help="")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()