# RAG and DSPy-Based Chatbot

## Overview

This project explores **Retrieval-Augmented Generation (RAG)** and **DSPy** through a document-based question-answering chatbot.

The implementation begins with a baseline RAG pipeline that retrieves relevant information from a collection of documents and uses a Large Language Model (LLM) to generate answers. It then extends the workflow with DSPy to structure the LLM task and experiment with programmatic optimisation.

The project compares the baseline and DSPy-based approaches using Semantic F1 against reference answers.

## Project Objectives

- Understand the end-to-end RAG workflow.
- Implement document chunking, embedding generation and vector similarity retrieval.
- Build a document-grounded chatbot using FAISS and an LLM.
- Explore DSPy Signatures, Modules and optimisation.
- Evaluate generated answers against reference answers.
- Compare baseline RAG and DSPy-optimised RAG.

## Technologies

- Python
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS
- Hugging Face Transformers
- Qwen2.5-3B-Instruct
- DSPy
- Gradio
- Pandas, NumPy and scikit-learn
- Google Colab

## Understanding the Approach

### 1. Retrieval-Augmented Generation (RAG)

RAG combines information retrieval with text generation. Instead of relying only on information encoded in an LLM, the system retrieves relevant passages from an external knowledge source and supplies them as context to the model.

The baseline pipeline follows these steps:

1. Load the source documents.
2. Split the documents into smaller text chunks.
3. Convert chunks into vector embeddings using Sentence Transformers.
4. Build a FAISS index for similarity search.
5. Retrieve relevant chunks for a user's question.
6. Pass the retrieved context and question to the LLM to generate an answer.

### 2. Why DSPy with RAG?

A conventional RAG implementation often relies on manually written prompts to guide the LLM. **DSPy (Declarative Self-improving Language Programs)** provides a programming framework for defining LLM tasks with structured Signatures and Modules, and for experimenting with optimisation against an evaluation metric.

In this project, DSPy is used to organise and optimise the question-answering part of the RAG workflow. It does not replace document chunking, embeddings or FAISS retrieval.

Key DSPy concepts explored:

- **Signatures:** Specify the inputs and expected outputs of an LLM task.
- **Modules:** Organise the task into reusable components.
- **Optimisers:** Use examples and a metric to produce and select demonstrations for a program.
- **Evaluation:** Measure the quality of generated answers against reference answers.

### 3. Baseline RAG vs DSPy + RAG

| Component | Baseline RAG | DSPy + RAG |
|---|---|---|
| Document processing | Yes | Yes |
| Embeddings | Sentence Transformers | Sentence Transformers |
| Vector retrieval | FAISS | FAISS |
| Language model | Qwen2.5-3B-Instruct | Qwen2.5-3B-Instruct |
| Task definition | Manually constructed prompt | DSPy Signature |
| Workflow | Retrieval and generation | DSPy RAG module |
| Optimisation | No DSPy optimisation | BootstrapFewShot |
| Evaluation | Baseline evaluation | Before/after comparison |

## Notebooks

### `notebooks/01_basic_rag.ipynb`

Implements the baseline RAG pipeline:

- Loads three text documents about RAG fundamentals, indexes and vector databases.
- Splits the documents into chunks.
- Generates embeddings using `all-MiniLM-L6-v2`.
- Creates a FAISS index containing 29 document embeddings.
- Retrieves the most relevant chunks for a query.
- Generates answers using Qwen2.5-3B-Instruct.
- Evaluates responses against a set of questions and reference answers.

### `notebooks/02_dspy_rag.ipynb`

Extends the RAG workflow using DSPy:

- Reuses the document processing, embedding and FAISS retrieval workflow.
- Connects Qwen2.5-3B-Instruct to DSPy.
- Defines a DSPy Signature and a RAG module using ChainOfThought.
- Uses 25 evaluation questions, with 15 training examples and 10 held-out test questions.
- Applies the BootstrapFewShot optimiser.
- Compares response quality before and after optimisation using Semantic F1.
- Includes a Gradio interface for chatbot interaction.

## Evaluation

The DSPy notebook reports the following average Semantic F1 scores on its held-out test set:

| Configuration | Average Semantic F1 |
|---|---:|
| Before DSPy optimisation | 0.8636 |
| After DSPy optimisation | 0.8312 |

In this experiment, optimisation did not increase the average Semantic F1 score. This is an important finding: optimisation does not guarantee better results. Outcomes can depend on the examples, metric, optimiser configuration and model behaviour.

The comparison is an experiment on a small evaluation set, not a general performance benchmark.

## Repository Structure

```text
RAG-DSPy-Chatbot/
├── notebooks/
│   ├── 01_basic_rag.ipynb
│   └── 02_dspy_rag.ipynb
├── data/
│   ├── documents/
│   │   ├── rag_fundamentals.txt
│   │   ├── rag_indexes.txt
│   │   └── rag_vector_databases.txt
│   └── evaluation_questions.csv
├── results/
│   └── rag_results.csv
├── README.md
├── requirements.txt
└── .gitignore
```

## Running the Notebooks

The notebooks were developed in Google Colab.

1. Clone or download this repository.
2. Open the notebooks in Google Colab.
3. Make the `data` directory available to the notebook, or update the file paths to match your environment.
4. Install the dependencies listed in `requirements.txt`.
5. Select a GPU-enabled runtime with sufficient memory for the language model.
6. Run `01_basic_rag.ipynb` first, followed by `02_dspy_rag.ipynb`.

The Qwen model is loaded through Hugging Face Transformers during notebook execution. PyTorch must be installed with a build compatible with the runtime's CUDA configuration.

## Scope and Limitations

- This is a learning and experimentation project developed in Google Colab.
- The knowledge base contains a small collection of documents.
- The evaluation uses 25 questions, with 10 held out for testing in the DSPy experiment.
- Results depend on the source documents, retrieval quality, model behaviour and evaluation setup.
- A higher Semantic F1 score on this small test set would not by itself establish general superiority.
---

This repository presents an independent learning and experimentation project focused on RAG, DSPy and LLM evaluation.
