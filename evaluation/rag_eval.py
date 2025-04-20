import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from rag import RAGAgent
from dotenv import load_dotenv
import openai
import time
import random

def safe_process_query(question, max_retries=5):
    for attempt in range(max_retries):
        try:
            return agent.process_query(question)
        except openai.RateLimitError as e:
            # exponential backoff + jitter
            backoff = (2 ** attempt) + random.uniform(0, 1)
            print(f"[RateLimit] attempt {attempt+1}/{max_retries}, sleeping {backoff:.1f}s…")
            time.sleep(backoff)
    # if exhaust retries, re‑raise
    raise RuntimeError(f"Failed after {max_retries} retries due to rate limits")

test_samples = [
    {   
        "question": "What was Apple’s revenue in Q1 2024? Just give me the number.",
        "ground_truth": "$90,753 million"
        },
    {
        "question": "What was Apple's total revenue in Q2 2024? Just give me the number.",
        "ground_truth": "$90,753 million"
    },
       {
        "question": "Which product category had the highest net sales in fiscal Q2 2024, and what was the amount?", 
        "ground_truth": "Services, 23867 million"
    },
        {
        "question": "What was Apple’s diluted earnings per share in fiscal Q2 2024? Just give me the number.",
        "ground_truth": "$1.53 per share"
    },
    {
        "question": "What was Apple's net sales by reportable segment for 2024?",
        "ground_truth":  "Americas: $167,045, Europe: $101,328, Greater China: $66,952, Japan: $25,052, Rest of Asia Pacific: $30,658"
    },
    {
        "question": "What was Apple's net sales by category for 2024?",
        "ground_truth": "iphone: $201,183, Mac: $29,984, ipad: 26,694, Wearables, Home and Accessories: $37,005, Services: $96,169" 
    },
        {
        "question": "What was better than expected in the quarter for service momentum?",
        "ground_truth": "In services, we've seen a very strong performance across the board. We've mentioned, we've had records in several categories, in several geographic segments. It's very broad based, our subscription business is going well. Transacting accounts and paid accounts are growing double-digits. And also we've seen a really strong performance both in developed and emerging markets. So very pleased with the way the services business is going."  
    },
    {
        "question": "In earnings call, what were the drivers of Apple’s acceleration in services growth from 11% to 14%, and why isn’t that pace more sustainable next quarter?",
        "ground_truth": "The acceleration was driven by all‑time records across both developed and emerging markets, with especially strong contributions from smaller but fast‑growing categories like cloud, video, and payment services. However, tougher year‑over‑year comparisons in the latter half of the fiscal year mean growth is expected to average the first‑half rate rather than continue accelerating." 
    },
    {
        "question": "What early feedback did Apple receive on Apple Intelligence and did Tim Cook attribute it to strong iPhone performance?",
        "ground_truth": "Within the first three days of the iOS 18.1 launch, adoption in US English was twice as fast as iOS 17.1 last year, with positive feedback from both developers and customers on features like systemwide Writing Tools, a more conversational Siri, intelligent Photos (including text‑driven movie creation), notification summaries, and email summaries/priority. While Cook highlighted this strong interest and upcoming December and 2025 feature rollouts (e.g., expanded Writing Tools, visual intelligence, ChatGPT integration, new languages), he did not explicitly tie Apple Intelligence to the quarter’s strong iPhone performance."
    },
        {
        "question": "From earnings call, could Apple’s combined revenue from other emerging markets soon surpass the $70 billion Greater China segment, and should investors view those markets as the next growth drivers?",
        "ground_truth": "China remains Apple’s largest emerging market, but other regions—India, Saudi Arabia, Mexico, Turkey, Brazil, and Indonesia—are seeing rapid growth thanks to large, low‑share populations and strong brand excitement. Collectively, their revenues are rising and closing the gap on China’s segment, and Apple expects that trajectory to continue."
    },

]

load_dotenv()
agent = RAGAgent(model_name="gpt-4o")

dataset_dict = {"question": [], "ground_truth": [], "contexts": [], "answer": []}

for sample in test_samples:
    q = sample["question"]
    gt = sample["ground_truth"]

    out = safe_process_query(q)
    contexts = [d["content"] for d in out["source_documents"]]

    dataset_dict["question"].append(q)
    dataset_dict["ground_truth"].append(gt)
    dataset_dict["contexts"].append(contexts)
    dataset_dict["answer"].append(out["response"])

    time.sleep(0.5)

dataset = Dataset.from_dict(dataset_dict)

result = evaluate(
    dataset=dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

df = result.to_pandas()

out_dir = "output"
os.makedirs(out_dir, exist_ok=True)

file_path = os.path.join(out_dir, "apple_rag_results.csv")

df.to_csv(file_path, index=False)


print(f"Saved results → {file_path}")
print(df)