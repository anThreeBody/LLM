import json
from sentence_transformers import SentenceTransformer, util

def get_sentence_encoder():
    model = "all-MiniLM-L6-v2"
    return SentenceTransformer(model)

encoder = get_sentence_encoder()
dataset_path = "demos\svamp/202404031059\demo_dataset.json"
demo =[]
with open(dataset_path, 'r',encoding="utf-8") as f:
        demo=json.load(f)

for i in range(len(demo)):
    samples = demo[i]['train_samples']
    for j in range(len(samples)):
        question =  samples[j]['question']
        embedding = encoder.encode(question)
        samples[j]["embedding"] = embedding.tolist()


dataset_embedding_path = "./demos/svamp/demo_embedding.json"
with open(dataset_embedding_path, 'w',encoding="utf-8") as f:
     json.dump(demo,f,indent=4)