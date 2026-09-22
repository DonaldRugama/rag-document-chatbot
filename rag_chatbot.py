"""A no-paid-API RAG chatbot for PDF, DOCX, and TXT documents."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
import gradio as gr
import numpy as np
import torch
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4


@dataclass
class Chunk:
    text: str
    source: str
    location: str


class RAGChatbot:
    def __init__(self) -> None:
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.chunks: list[Chunk] = []
        self.tokenizer = None
        self.model = None

    @staticmethod
    def _sliding_chunks(text: str) -> Iterable[str]:
        cleaned = " ".join(text.split())
        if not cleaned:
            return
        start = 0
        while start < len(cleaned):
            end = min(start + CHUNK_SIZE, len(cleaned))
            if end < len(cleaned):
                boundary = cleaned.rfind(". ", start, end)
                if boundary > start + CHUNK_SIZE // 2:
                    end = boundary + 1
            yield cleaned[start:end].strip()
            if end == len(cleaned):
                break
            start = max(end - CHUNK_OVERLAP, start + 1)

    def _extract_file(self, file_path: str) -> list[Chunk]:
        path = Path(file_path)
        suffix = path.suffix.lower()
        chunks: list[Chunk] = []

        if suffix == ".pdf":
            for page_number, page in enumerate(PdfReader(path).pages, start=1):
                for text in self._sliding_chunks(page.extract_text() or ""):
                    chunks.append(Chunk(text, path.name, f"page {page_number}"))
        elif suffix == ".docx":
            text = "\n".join(p.text for p in Document(path).paragraphs if p.text.strip())
            for number, part in enumerate(self._sliding_chunks(text), start=1):
                chunks.append(Chunk(part, path.name, f"section {number}"))
        elif suffix == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
            for number, part in enumerate(self._sliding_chunks(text), start=1):
                chunks.append(Chunk(part, path.name, f"section {number}"))
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Use PDF, DOCX, or TXT.")
        return chunks

    def index_files(self, files: list[str] | None) -> str:
        if not files:
            return "Please upload at least one PDF, DOCX, or TXT file."

        new_chunks: list[Chunk] = []
        errors: list[str] = []
        for file_path in files:
            try:
                new_chunks.extend(self._extract_file(file_path))
            except Exception as exc:
                errors.append(f"{Path(file_path).name}: {exc}")

        if not new_chunks:
            return "No readable text was found. " + " ".join(errors)

        embeddings = self.embedder.encode(
            [chunk.text for chunk in new_chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.chunks = new_chunks
        names = sorted({chunk.source for chunk in new_chunks})
        status = f"Indexed {len(new_chunks)} chunks from {len(names)} document(s): {', '.join(names)}."
        if errors:
            status += " Skipped: " + " | ".join(errors)
        return status

    def retrieve(self, question: str) -> list[tuple[Chunk, float]]:
        if self.index is None:
            return []
        query = self.embedder.encode(
            [question], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(query, min(TOP_K, len(self.chunks)))
        return [(self.chunks[i], float(score)) for i, score in zip(indices[0], scores[0])]

    def _load_generator(self) -> None:
        if self.model is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            GENERATION_MODEL,
            torch_dtype="auto",
            device_map="auto",
        )

    def answer(self, question: str, history: list | None = None) -> str:
        matches = self.retrieve(question)
        if not matches:
            return "Please upload your documents and click **Build knowledge base** first."

        self._load_generator()
        context_parts = []
        sources = []
        for number, (chunk, score) in enumerate(matches, start=1):
            label = f"{chunk.source}, {chunk.location}"
            context_parts.append(f"SOURCE {number} [{label}]\n{chunk.text}")
            sources.append(f"- [{label}] — similarity {score:.2f}")

        recent_history = ""
        if history:
            compact = history[-3:]
            recent_history = "\nRecent conversation: " + str(compact)

        system_message = (
            "You answer questions only from the supplied document excerpts. "
            "If the answer is not supported by them, say: "
            "'I could not find that in the uploaded documents.' "
            "Be concise, and cite supporting excerpts as [Source 1], [Source 2], etc."
        )
        user_message = (
            f"DOCUMENT EXCERPTS:\n{'\n\n'.join(context_parts)}"
            f"{recent_history}\n\nQUESTION: {question}"
        )
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **model_inputs,
                max_new_tokens=250,
                do_sample=False,
                repetition_penalty=1.05,
            )
        generated = output[0][model_inputs.input_ids.shape[1] :]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return answer + "\n\n**Retrieved sources**\n" + "\n".join(sources)


bot = RAGChatbot()

with gr.Blocks(title="Document RAG Chatbot") as demo:
    gr.Markdown(
        "# Document RAG Chatbot\n"
        "Upload custom documents, build the vector index, and ask grounded questions. "
        "The app uses open-source models and does not require a paid API key."
    )
    with gr.Row():
        files = gr.File(
            label="Documents",
            file_count="multiple",
            file_types=[".pdf", ".docx", ".txt"],
            type="filepath",
        )
        with gr.Column():
            build_button = gr.Button("Build knowledge base", variant="primary")
            status = gr.Markdown("No documents indexed yet.")
    build_button.click(bot.index_files, inputs=files, outputs=status)
    gr.ChatInterface(
        fn=bot.answer,
        title="Ask your documents",
        examples=[
            "Summarize the main ideas.",
            "What conclusions does the document make?",
            "Which source supports that answer?",
        ],
    )

if __name__ == "__main__":
    demo.launch(debug=True)
