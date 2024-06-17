# LLM
针对当前思维链提示技术缺乏进一步人类策略的指导和对小模型应用效果不佳的问题，提出了一种基于6W2H问题分解策略的思维链提示框架WH-CoT（6W2H Chain-of-Thought）。
首先利用Sentence-BERT模型对任务数据集进行聚类采样，划分为训练集和测试集；
接着在训练集中对所有样本进行元素提取、问题分解、构建答案段落和生成答案等操作形成思维链，进而构建任务语料库；
最后在推理阶段，自适应地从语料库中采样演示样本添加到提示词中，模型结合提示词生成测试问题的答案。
对于Qwen-turbo模型，在算术推理任务上，WH-CoT的平均准确率相较于主流的Zero-Shot-CoT和Manual-CoT，分别提升了3.35和4.27个百分点；
在多跳推理任务上，WH-CoT的总性能提升比在EM上相较于Zero-Shot-CoT和Manual-CoT分别提升了36和111个百分点。
另外，对于中小规模的Qwen-14B-Chat和Qwen-7B-Chat模型，WH-CoT的总性能提升比在EM和F1上均高于Zero-Shot-CoT和Manual-CoT。
WH-CoT将人类策略与机器智能进一步相结合，对不同规模的大语言模型，均能有效地提升了其在算术推理和多跳推理任务上的推理性能。

运行评估代码：（需先设置自己的qwen的api-key）
python eval.py --model qwen-turbo  --task gsm8k --demo_num 3 --method wh-cot --sample cluster
