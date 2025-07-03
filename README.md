# Acknoledgement
This is a fork of Google Gemini quick start project.
If you are looking for official examples, use the original repo!

This project is in order to running AI locally, and can support commodity hardware e.g. I have GTX 1070 Ti.

This project doesn't incldue tests at this point and not intended for production or critical usage, it's main goal is personal AI usage.

Enjoy!

# LangGraph Quickstart

This project demonstrates a fullstack application using a React frontend and a LangGraph-powered backend agent. The agent is designed to perform comprehensive research on a user's query by dynamically generating search terms, querying the web using Google Search, reflecting on the results to identify knowledge gaps, and iteratively refining its search until it can provide a well-supported answer with citations. This application serves as an example of building research-augmented conversational AI using OpenAI API-compatible models and other tools e.g. DuckDuckGo.

![Fullstack LangGraph](./app.png)

## Features

- 💬 Fullstack application with a React frontend and LangGraph backend.
- 🧠 Powered by a LangGraph agent for advanced research and conversational AI.
- 🔍 Dynamic search query generation using Google Gemini models.
- 🌐 Integrated web research via DuckDuckGo API.
- 🤔 Reflective reasoning to identify knowledge gaps and refine searches.
- 📄 Generates answers with citations from gathered sources.
- 🔄 Hot-reloading for both frontend and backend development during development.

## Project Structure

The project is divided into two main directories:

-   `frontend/`: Contains the React application built with Vite.
-   `backend/`: Contains the LangGraph/FastAPI application, including the research agent logic.
-   `services/`: Contains the Services layer, including OpenAI APIs etc.

## Getting Started: Development and Local Testing

Follow these steps to get the application running locally for development and testing.

**1. Prerequisites:**

-   Node.js and npm (or yarn/pnpm)
-   Python 3.12+
-   virtualenv -p python3.12 .venv

**2. Install Dependencies:**

**Backend:**

```bash
cd backend
uv pip install -e '.[dev]'
uv build
```

**Frontend:**

```bash
cd frontend
npm install
```

**Services:**

```bash
cd services
uv pip install -e '.[dev]' --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
# You may need to install the latest transformer library: uv pip install git+https://github.com/huggingface/transformers.git
uv build
```

**3. Run Development Servers:**

**Services:**
This are optional, and only needed if you want to host models directly from HuggingFace.

```bash
cd services
make dev
```
This will run the uvicorn development servers.    Open your browser and navigate to the frontend development server URL (e.g., `http://127.0.0.1/docs`).

You will need to download and cache the models first time, and you need to have a huggingface account and token.
And then set the environment variables:

```bash
export HF_HOME:  $(pwd)/../models
export HF_TOKEN: <YOU TOKEN>
mkdir $HF_HOME 
# expecting models director at the same level as "services"
```

The simplest way is to run the test:
```bash
pytest -vv .\tests\test_registry.py
```
This will take some time, make sure you have sufficient hard disk space.

**Backend:**
For the backend, open a terminal in the `backend/` directory and run `langgraph dev` or:
```bash
cd backend
make dev
```
The backend API will be available at `http://127.0.0.1:2024` and the debugger or `http://127.0.0.1::2025`.

**Frontend:**
For the frontend, open a terminal in the `frontend/` directory and run `npm run dev` or:
```bash
cd frontend
make dev
```
This will run the frontend development servers.    Open your browser and navigate to the frontend development server URL (e.g., `http://localhost:5173/app`).


## How the Backend Agent Works (High-Level)

The core of the backend is a LangGraph agent defined in `backend/src/agent/graph.py`. It follows these steps:

![Agent Flow](./agent.png)

1.  **Generate Initial Queries:** Based on your input, it generates a set of initial search queries using a OpenAI API-compatible model (e.g. Qwen3).
2.  **Web Research:** For each query, it uses the Gemini model with the DuckDuckGo tool to find relevant web pages.
3.  **Reflection & Knowledge Gap Analysis:** The agent analyzes the search results to determine if the information is sufficient or if there are knowledge gaps. It uses a OpenAI API-compatible model (e.g. Qwen3) for this reflection process.
4.  **Iterative Refinement:** If gaps are found or the information is insufficient, it generates follow-up queries and repeats the web research and reflection steps (up to a configured maximum number of loops).
5.  **Finalize Answer:** Once the research is deemed sufficient, the agent synthesizes the gathered information into a coherent answer, including citations from the web sources, using OpenAI API-compatible model (e.g. Qwen3).

## Deployment

In production, the backend server serves the optimized static frontend build. LangGraph requires a Redis instance and a Postgres database. Redis is used as a pub-sub broker to enable streaming real time output from background runs. Postgres is used to store assistants, threads, runs, persist thread state and long term memory, and to manage the state of the background task queue with 'exactly once' semantics. For more details on how to deploy the backend server, take a look at the [LangGraph Documentation](https://langchain-ai.github.io/langgraph/concepts/deployment_options/). 

## Technologies Used

- [React](https://reactjs.org/) (with [Vite](https://vitejs.dev/)) - For the frontend user interface.
- [Tailwind CSS](https://tailwindcss.com/) - For styling.
- [Shadcn UI](https://ui.shadcn.com/) - For components.
- [LangGraph](https://github.com/langchain-ai/langgraph) - For building the backend research agent.
- [DuckDuckGo Gemini](https://python.langchain.com/docs/integrations/tools/ddg/) - tool for web searches.
- [LM Studio](https://lmstudio.ai/) - for providing local inference to models sourced from HuggingFace (GGUF) and providing OpenAI API-compatible interface.
- [Qwen3](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f) - OSS models that support reasoning and tool usage, sourced from HuggingFace.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details. 