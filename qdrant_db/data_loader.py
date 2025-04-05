from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import cohere
from cohere.core.api_error import ApiError
import time
import glob
import os
import re
import requests

# wait for qdrant db to be ready
qdrant_url = "http://qdrant:6333/healthz"
print("Waiting for Qdrant to be ready...")

while True:
    try:
        response = requests.get(qdrant_url)
        if response.status_code == 200:
            print("Qdrant is up!")
            break
    except requests.exceptions.ConnectionError:
        pass
    time.sleep(1)

# initalise embedding model
load_dotenv()
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
co = cohere.ClientV2(COHERE_API_KEY)
model="embed-english-v3.0"
VECTOR_SIZE = 1024
chunk_size = 3000

# connect to qdrant
client = QdrantClient("qdrant", port=6333)

# process data
def process_data(data_path, collection_name, check_files = False):
    # create collection if it doesn't exist
    try:
        client.get_collection(collection_name)
        print(f"Collection {collection_name} already exists")
    except:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)
        )
        print(f"Created collection {collection_name}")

    # process each file
    files = glob.glob(data_path)
    print(f"Found {len(files)} files to process for {collection_name}")

    for file_path in files:
        try:
            file_name = os.path.basename(file_path)
            
            if check_files:
                #check if file was already processed
                search_results = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=models.Filter(
                        must=[models.FieldCondition(key="file_name", match=models.MatchValue(value=file_name))]
                    ),
                    limit=1
                )
                
                if search_results:
                    print(f"Skipping already processed file: {file_name}")
                    continue
                
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
            print(f"Processing {file_name} - split into {len(chunks)} chunks")

            doc_embeddings = co.embed(texts=chunks,
                                    model=model,
                                    input_type="search_document",
                                    embedding_types=['float'])
            
            points = []
            for idx, (embedding, doc) in enumerate(zip(doc_embeddings.embeddings.float_, chunks)):
                # Create a unique ID that includes the filename to avoid conflicts
                unique_id = f"{file_name}_{idx}"
                # Convert string ID to integer using hash
                numeric_id = hash(unique_id) % (2**63 - 1)
                point = models.PointStruct(
                    id=numeric_id,
                    vector=embedding,
                    payload={
                            "document": doc,
                            "file_name": file_name,
                            "chunk_index": idx
                        }
                )
                points.append(point)
            
            # upload to Qdrant
            client.upsert(
                collection_name=collection_name,
                points=points
            )

            print(f"Processed {file_name}")
        except ApiError as e:
            if e.status_code == 429:  # rate limit exceeded error
                backoff_time = 60
                print(f"Rate limit exceeded, retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)  # wait before retrying
            else:
                print(f"An error occurred: {e.message}")

# process news data
LINK_PATTERN = re.compile(r"^link:\s*(.+)", re.MULTILINE)
SOURCE_PATTERN = re.compile(r"^source:\s*(.+)", re.MULTILINE)
TITLE_PATTERN = re.compile(r"^title:\s*(.+)", re.MULTILINE)
SUMMARY_PATTERN = re.compile(r"^summary:\s*(.*)", re.MULTILINE)
DATE_PATTERN = re.compile(r"^publishDate:\s*(.+)", re.MULTILINE)
CONTENT_PATTERN = re.compile(r"^content:\s*(.+)", re.MULTILINE)

def process_news_data(data_path, collection_name, check_files = False):
    # create collection if it doesn't exist
    try:
        client.get_collection(collection_name)
        print(f"Collection {collection_name} already exists")
    except:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)
        )
        print(f"Created collection {collection_name}")
    
    # process each file
    files = glob.glob(data_path)
    print(f"Found {len(files)} files to process for {collection_name}")

    for file_path in files:
        try:
            file_name = os.path.basename(file_path)
            
            if check_files:
                # check if file was already processed
                search_results = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=models.Filter(
                        must=[models.FieldCondition(key="file_name", match=models.MatchValue(value=file_name))]
                    ),
                    limit=1
                )
                
                if search_results:
                    print(f"Skipping already processed file: {file_name}")
                    continue
                
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Extract metadata using regex
            link = LINK_PATTERN.search(content).group(1) if LINK_PATTERN.search(content) else ""
            source = SOURCE_PATTERN.search(content).group(1) if SOURCE_PATTERN.search(content) else "Unknown"
            title = TITLE_PATTERN.search(content).group(1) if TITLE_PATTERN.search(content) else file_name.replace(".txt", "")
            summary = SUMMARY_PATTERN.search(content).group(1) if SUMMARY_PATTERN.search(content) else ""
            publish_date = DATE_PATTERN.search(content).group(1) if DATE_PATTERN.search(content) else "2025-03-31T12:00:00Z"
            
            # Extract and clean content (remove metadata lines)
            content_match = CONTENT_PATTERN.search(content)
            cleaned_content = content_match.group(1).strip() if content_match else ""

            # generate summary
            summary = cleaned_content[:200] + "..." if len(cleaned_content) > 200 else cleaned_content
            
            chunks = [cleaned_content[i:i+chunk_size] for i in range(0, len(cleaned_content), chunk_size)]
            print(f"Processing {file_name} - split into {len(chunks)} chunks")

            doc_embeddings = co.embed(texts=chunks,
                                    model=model,
                                    input_type="search_document",
                                    embedding_types=['float'])

            points = []
            for idx, (embedding, doc) in enumerate(zip(doc_embeddings.embeddings.float_, chunks)):
                # Create a unique ID that includes the filename to avoid conflicts
                unique_id = f"{file_name}_{idx}"
                # Convert string ID to integer using hash
                numeric_id = hash(unique_id) % (2**63 - 1)
                point = models.PointStruct(
                    id=numeric_id,
                    vector=embedding,
                    payload={
                            "document": doc,
                            "file_name": file_name,
                            "chunk_index": idx,
                            "title": title,
                            "source": source,
                            "summary": summary,
                            "publish_date": publish_date,
                            "url": link,
                        }
                )
                points.append(point)
            
            # upload to Qdrant
            client.upsert(
                collection_name=collection_name,
                points=points
            )

            print(f"Processed {file_name}")

        except ApiError as e:
            if e.status_code == 429:  # rate limit exceeded error
                backoff_time = 60
                for i in range(5):  # Retry 5 times with exponential backoff
                    print(f"Rate limit exceeded, retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                    backoff_time *= 2  # Exponential backoff
            else:
                print(f"An error occurred: {e.message}")


if __name__ == "__main__":
    print("Starting data loading process...")
    process_data(data_path="/data/aapl_10k_10Q_forms/apple_filings_text_10K/*.txt", collection_name="aapl_10k_10q_forms", check_files=False)
    process_data(data_path="/data/aapl_10k_10Q_forms/apple_filings_text_10Q/*.txt", collection_name="aapl_10k_10q_forms", check_files=False)
    process_data(data_path="/data/earnings_calls/earnings_calls_txt/*.txt", collection_name="earnings_calls", check_files=False)
    process_news_data(data_path="/data/financial_news/articles/*.txt", collection_name="financial_news", check_files=False)
    print("Data loading complete!")


