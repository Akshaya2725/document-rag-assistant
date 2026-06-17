# Document Retrieval-Augmented Generation (RAG) System

This repository provides a modular, document-agnostic Retrieval-Augmented Generation (RAG) architecture developed for semantic text extraction, localized indexing, and context-grounded document analysis.

The script is optimized to deploy directly inside a Google Colab notebook workspace, enabling an end-to-end interactive document chat pipeline.

The architecture dynamically handles arbitrary Portable Document Format (PDF) files by combining local transformer-based semantic token embedding layers with remote frontier large language models through an OpenAI client wrapper.

---

# Architecture and Execution Pipeline

The script organizes runtime processes into a sequential, decoupled data engineering workflow:

## 1. In-Memory Storage Authentication

Authenticates securely via local notebook management abstractions, verifying local environment tokens before initiating client connections.

## 2. Interactive Document Ingestion

Accepts arbitrary user documents through an asynchronous file upload stream. The structural PDF binary layer is processed and parsed linearly using `PyPDFLoader`.

## 3. Contextual Text Segmentation

Text blocks are managed by a `RecursiveCharacterTextSplitter` algorithm configured with:

* Chunk Size: `1000` characters
* Chunk Overlap: `200` characters

This mechanism maintains semantic continuity across text segment boundaries and optimizes downstream token payload generation.

## 4. Local Vectorization and Indexing

Tokenized text chunks are transformed into dense 384-dimensional vector embeddings completely locally using the `all-MiniLM-L6-v2` Hugging Face transformer model.

Computed embedding vectors are indexed into a local runtime-managed `ChromaDB` vector database instance.

## 5. Context Retrieval

User queries execute a similarity-mapping vector search operation.

The system isolates the top five (`k = 5`) most relevant vector neighborhoods to construct a diverse and contextually robust reference block.

## 6. Unified API Inference

The retrieved context block is merged into a system template layer that enforces strict contextual grounding.

The resulting payload is securely dispatched through the Bluesminds server routing layer:

```text
https://api.bluesminds.com/v1
```

using the `gpt-5.2-chat` model engine.

---

# Repository Layout

```text
rag-document-assistant/
│
├── document_rag_assistant.ipynb            # Complete application source code optimized for Google Colab
├── requirements.txt   # Explicit dependency listing for version alignment
└── README.md          # Technical pipeline documentation
```

---

# Setup and Runtime Instructions

## 1. Environment Credentialing

The code enforces the use of protected environment variables to mitigate API credential leakage.

### Steps

1. Open your notebook environment in Google Colab.
2. Select the **Secrets** tab (represented by the key icon in the left-hand navigation interface).
3. Create a new secret with the following configuration:

```text
Name  : OPENAI_API_KEY
Value : [Your personal Bluesminds API credential token]
```

4. Enable notebook visibility access for this secret.

---

## 2. Dependency Management

Verify that your workspace contains the exact software dependencies cataloged within `requirements.txt`.

```text
langchain==0.3.0
langchain-classic==0.1.0
langchain-community==0.3.0
langchain-openai==0.2.0
chromadb==0.5.0
pypdf==4.2.0
sentence-transformers==3.0.1
numpy
```

---

## 3. Execution Procedure

Execute the cell containing `document_rag_assistant.ipynb` inside the Google Colab notebook.

### Step 1

The interface presents a file upload widget.

Upload your target PDF document.

### Steps 2–4

The terminal interface outputs system validation metrics, including:

* Document parsing statistics
* Contextual text chunk extraction details
* Local vector database indexing status
* API connection verification logs

### Step 5

An interactive chat console opens directly within the notebook output window.

Enter questions related to the uploaded document through the prompt interface.

To terminate the session securely, type:

```text
end
```

The application will close the active chat loop and release the API connection.

---

# Features

* Document-agnostic PDF ingestion
* Semantic chunking with overlap preservation
* Local transformer-based embeddings
* ChromaDB vector indexing
* Similarity-based contextual retrieval
* Context-grounded response generation
* Google Colab optimized workflow
* Secure environment-based credential management

---

# Use Cases

* Research paper analysis
* Technical document exploration
* Enterprise knowledge retrieval
* Educational content querying
* Contract and policy document review
* Context-aware document question answering


# Sample Output

## Document Tested

* **Book:** *Crime and Punishment*
* **Length:** 967 Pages

## Evaluation Queries

The uploaded PDF was successfully processed, indexed, and stored within the local ChromaDB vector database. The Retrieval-Augmented Generation (RAG) pipeline was then evaluated using the following document-grounded queries:

1. What is the name of the book?
2. How many chapters are present in the book?
3. Which characters are mentioned in the book?
4. What is the story about?

## Result

The system successfully retrieved semantically relevant context from the uploaded document and generated accurate responses grounded in the source material.

The answers demonstrated that the retrieval layer correctly identified relevant document sections while the language model produced coherent, context-aware responses based solely on the retrieved content.

### Example Queries

```text
Query: What is the name of the book?
Answer: Crime and Punishment

Query: How many chapters are present in the book?
Answer: [Generated from retrieved context]

Query: Which characters are mentioned in the book?
Answer: [Generated from retrieved context]

Query: What is the story about?
Answer: [Generated from retrieved context]
```

### PDF Upload and Processing

![PDF Upload](images/img1.png)

### Document Chunking and Indexing

![Chunking](images/img2.png)

### Vector Database Creation

![Vector Database](images/img3.png)

### Query 1: Book Name

![Book Name Query](images/img4.png)

### Query 2: Total Chapters

![Chapter Query](images/img5.png)

### Query 3: Characters Mentioned

![Characters Query](images/img6.png)

### Query 4: Story Summary

![Story Query](images/img7.png)
