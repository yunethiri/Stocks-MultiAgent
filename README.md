# Stocks Multi-Agent

This project is a multi-agent for analyzing Apple stock data using a combination of specialized agents. These agents perform tasks such as intent classification, sentiment analysis, document retrieval, and visualization. The system is containerized using Docker and orchestrated with Docker Compose.

## Environment Variables

Before running the project, create a `.env` file in the **agents** and **qdrant_db** folder. This file should include your OPENAI_API_KEY, COHERE_API_KEY and BRAVE_AI_API_KEY.

> **Important:**  
> - Do **not** commit the `.env` file to version control.  
> - It must reside in the `agents/` and `qdrant_db` folder so that the containers can load it during build/runtime.

### Additional Frontend Configuration
The Streamlit frontend also requires access to your OpenAI API key. You must provide this in the `frontend/.streamlit` directory as two configuration files:

- `config.toml` – for basic Streamlit settings
- `secrets.toml` – for securely passing secrets like API keys

**frontend/.streamlit/secrets.toml**
```toml
openai_api_key = "your-openai-api-key"
```
> This file should also **not** be committed to version control.

## Docker Architecture Overview

This project uses Docker Compose to build and run multiple containers that are interconnected by a shared internal network:

- **Backend Container:**  
  Contains the multi-agent system (FastAPI backend), which processes queries, performs sentiment and document analysis, and synthesizes final responses.
  
- **Frontend Container:**  
  Hosts the Streamlit application that serves as the user interface. The frontend communicates with the backend container via Requests.

- **Qdrant Database Container:**  
  Runs the Qdrant vector database for storing and retrieving data needed by the sentiment analysis and RAG agents.  
  - The Qdrant container uses persistent storage (`qdrant_data/`) to maintain its data.
  - It is accessible internally by the backend (e.g., via the hostname `qdrant` on port 6333).

> **Networking:**  
> All containers run on the same Docker network, allowing the frontend to connect to the backend (using the service name `backend`) and the backend to connect to the Qdrant database (using the service name `qdrant`).

## Getting the Project Running

Follow these steps to run the project:

1. **Clone the Repository:**  
   Clone or download the repository to your local machine.

2. **Set Up Environment Variables:**  
   In the `agents/` and `qdrant_db` folders, create a `.env` file containing your API keys (see above).
   
3. **Configure Frontend Secrets:**
   In the frontend/.streamlit/ directory, create the secrets.toml files with your OpenAI API key as shown above.

4. **Build and Start the Containers:**  
   From the project root, run:
   ```
   docker compose up -d
   ```

5. **Access the Frontend:**
    Navigate to `http://localhost:8501` in your web browser to access the Streamlit UI.

