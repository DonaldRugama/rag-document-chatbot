# No-Cost Document RAG Chatbot

This beginner-friendly portfolio project answers questions from custom PDF, DOCX, and TXT files. It performs the full Retrieval-Augmented Generation (RAG) process locally and does not require a paid API key.

> **Try it quickly:** Upload `sample_document.txt`, build the knowledge base, and ask one of the sample questions below. The included sample is fictional and was created specifically for this project.

## What the project demonstrates

- **Applied NLP:** document text extraction, cleaning, and overlapping chunk creation
- **Embeddings:** conversion of text chunks and questions into semantic vectors
- **Intelligent retrieval:** cosine-similarity search with a FAISS vector index
- **Generative AI:** an open-source instruction model writes an answer from retrieved context
- **Context-aware responses:** the latest chat turns are supplied to the model
- **Grounding and citations:** the prompt restricts answers to uploaded documents and lists retrieved pages or sections

## How RAG works here

1. The user uploads documents.
2. The program extracts and divides their text into overlapping chunks.
3. Sentence Transformers converts each chunk into an embedding.
4. FAISS stores the embeddings and retrieves chunks similar to a question.
5. Qwen receives only those retrieved chunks and generates a grounded response.

## Run free in Google Colab

1. Open a new notebook at [Google Colab](https://colab.research.google.com/).
2. Select **Runtime → Change runtime type → T4 GPU** if a free GPU is available.
3. Upload `rag_chatbot.py` and `requirements.txt` using Colab's Files panel.
4. Run these cells in order:

```python
!pip install -q -r requirements.txt
```

```python
%run rag_chatbot.py
```

5. Use the Gradio interface displayed below the cell, upload documents, and click **Build knowledge base**.

The first run downloads the open-source embedding and generation models, so it takes longer. Colab's free runtime and GPU availability are not guaranteed. The app can use CPU, but generation will be much slower.

## Run on a computer

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then run:

```bash
pip install -r requirements.txt
python rag_chatbot.py
```

## Project structure

```text
rag_document_chatbot/
├── rag_chatbot.py       # Complete RAG pipeline and Gradio interface
├── requirements.txt     # Python dependencies
├── sample_document.txt  # Original fictional document for testing
├── .gitignore           # Files Git should not publish
└── README.md            # Setup, concepts, and portfolio notes
```

## Test the chatbot

Upload `sample_document.txt`, click **Build knowledge base**, and try:

- What was Northstar Learning's total enrollment in 2026?
- Which course had the highest completion rate?
- What improvements are planned for 2027?
- Who is the CEO? *(The correct behavior is to say this is not in the document.)*

The final question tests grounding: the chatbot should not invent an answer when the information is absent.

## Models and libraries

- `sentence-transformers/all-MiniLM-L6-v2` creates embeddings.
- FAISS performs vector similarity search.
- `Qwen/Qwen2.5-1.5B-Instruct` generates answers without a paid API.
- Gradio provides the upload and chat interface.
- PyPDF and python-docx extract document text.

## Important limitations

- Scanned/image-only PDFs need OCR, which this starter version does not include.
- The small local model is free and practical, but less capable than large paid models.
- Colab sessions are temporary, so downloaded models and indexes disappear when the runtime resets.
- Do not upload confidential documents to a public or shared runtime.
- For a stronger evaluation, test questions whose answers are present and absent from the documents.

## Suggested portfolio description

> Built a RAG-powered document chatbot in Python using Sentence Transformers, FAISS, an open-source Qwen language model, and Gradio. Implemented document ingestion, overlapping chunking, semantic vector retrieval, context-grounded generation, conversation context, and source attribution for PDF, DOCX, and TXT files without relying on a paid AI API.

## Good next upgrades

- Add OCR for scanned PDFs.
- Save and reload the FAISS index.
- Add retrieval-quality evaluation and a small test set.
- Deploy the app to a free community hosting tier, subject to its current usage limits.
