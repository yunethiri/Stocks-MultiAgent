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

test_samples = [
    {   
        "question": "What was Apple’s revenue in Q1 2024?",
        "ground_truth": "$119.58 billion"
        },
    {
        "question": "What earnings per share (EPS) did Apple report for the December 2024 quarter?",
        "ground_truth": "$2.18"
    },
       {
        "question": "What was Apple’s net income for the December 2024 quarter?", 
        "ground_truth": "$33.9 billion"
    },
        {
        "question": "What guidance did Apple give for company gross margin in the March 2024 quarter?",
        "ground_truth": "between 46% and 47%"
    },
    {
        "question": "What were Apple’s total assets at September 28, 2024?",
        "ground_truth": "$364.98 billion" 
    },
    {
        "question": "What dividend per share did Apple declare in fiscal year 2024?",
        "ground_truth": "$0.98 per share or RSU" 
    },
        {
        "question": "What was Apple’s net income for fiscal year 2024?",
        "ground_truth": "$93.7 billion"  
    },
    {
        "question": "What was Apple’s basic earnings per share for fiscal year 2024?",
        "ground_truth": "$6.11" 
    },
    {
        "question": "What was Apple’s diluted earnings per share for fiscal year 2024?",
        "ground_truth": "$6.08"  
    },
        {
        "question": "How much did Apple spend on repurchasing its common stock in fiscal year 2024?",
        "ground_truth": "$95.8 billion"
    },

]

load_dotenv()
agent = RAGAgent(model_name="gpt-4o")

dataset_dict = {"question": [], "ground_truth": [], "contexts": [], "answer": []}

for sample in test_samples:
    q = sample["question"]
    gt = sample["ground_truth"]

    out = agent.process_query(q)
    # out["source_documents"] is a list of dicts with a "content" field:
    contexts = [d["content"] for d in out["source_documents"]]

    dataset_dict["question"].append(q)
    dataset_dict["ground_truth"].append(gt)
    dataset_dict["contexts"].append(contexts)
    dataset_dict["answer"].append(out["response"])

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