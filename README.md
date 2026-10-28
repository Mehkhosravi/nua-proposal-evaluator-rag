# NUA Proposal Evaluator RAG

A FastAPI retrieval-augmented generation service for querying New Urban Agenda
(NUA) and international development documents and assessing project proposals.
It stores PDF text chunks in ChromaDB and embeds them with
SentenceTransformers (`all-MiniLM-L6-v2`). OpenAI generates grounded answers
and evaluation reports.

## Requirements

- Docker and Docker Compose, or Python 3.11
- An OpenAI API key for `/query` and `/evaluate-proposal`
- Text-based PDF framework documents (scanned PDFs need OCR first)

## Run with Docker Compose

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
2. Put authorized framework PDFs in `data/`.
3. Build and start the API:

   ```sh
   docker compose up --build
   ```

The API is available at <http://localhost:8000>; interactive documentation is
at <http://localhost:8000/docs>. The `data/` and `chroma_db/` directories are
mounted into the container so documents and the persistent vector store remain
available between restarts.

## Run locally

```sh
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` in your shell (or load values from a local `.env` file),
add PDFs to `data/`, and start the server:

```sh
uvicorn main:app --reload
```

## Ingest documents

After adding PDFs, call ingestion once and again whenever the PDF collection
changes. Ingestion re-creates the collection from all PDFs currently under
`data/`.

```sh
curl -X POST http://localhost:8000/ingest
```

The response reports the number of PDFs processed and chunks created. Ingestion
does not require an OpenAI API key; embedding model weights are downloaded on
first use.

## Query the framework

```sh
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does the New Urban Agenda say about inclusive public spaces?",
    "k": 4
  }'
```

The response contains a grounded `answer` and the source filenames and page
numbers used. The endpoint returns an error if no documents have been ingested
or if the OpenAI API key is missing.

## Evaluate a project proposal

Send a JSON proposal with `title`, `summary`, `region`, and `activities`:

```sh
curl -X POST http://localhost:8000/evaluate-proposal \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Inclusive neighbourhood public spaces",
    "summary": "Improve equitable access to safe, accessible public spaces.",
    "region": "Example City",
    "activities": [
      "Co-design upgrades with residents, including underserved groups",
      "Improve accessibility, lighting, shade, and active mobility connections",
      "Track participation and public-space access outcomes"
    ]
  }'
```

The structured response includes a 0–100 alignment score and level, an overall
assessment, cited NUA principles, source page references, strengths, gaps, and
actionable recommendations. It evaluates only the proposal details and retrieved passages;
it is decision support, not a substitute for expert or community review.

## Endpoints

- `GET /health` — service status
- `POST /ingest` — ingest all PDFs from `data/`
- `POST /query` — ask a grounded NUA/development question
- `POST /evaluate-proposal` — retrieve NUA evidence and return an evaluation

Configuration can be overridden with `OPENAI_MODEL`, `EMBEDDING_MODEL`,
`DATA_DIR`, and `CHROMA_PERSIST_DIRECTORY`.
