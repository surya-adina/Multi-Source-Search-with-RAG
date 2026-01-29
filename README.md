<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

<img src="Multi-Source-Search-with-RAG.png" width="30%" style="position: relative; top: 0; right: 0;" alt="Project Logo"/>

# MULTI-SOURCE-SEARCH-WITH-RAG

<em>Unlock Knowledge Instantly with Smarter Search</em>

<!-- BADGES -->
<img src="https://img.shields.io/github/license/surya-adina/Multi-Source-Search-with-RAG?style=flat&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
<img src="https://img.shields.io/github/last-commit/surya-adina/Multi-Source-Search-with-RAG?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/surya-adina/Multi-Source-Search-with-RAG?style=flat&color=0080ff" alt="repo-top-language">
<img src="https://img.shields.io/github/languages/count/surya-adina/Multi-Source-Search-with-RAG?style=flat&color=0080ff" alt="repo-language-count">

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/Markdown-000000.svg?style=flat&logo=Markdown&logoColor=white" alt="Markdown">
<img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white" alt="Streamlit">
<img src="https://img.shields.io/badge/tqdm-FFC107.svg?style=flat&logo=tqdm&logoColor=black" alt="tqdm">
<img src="https://img.shields.io/badge/LangChain-1C3C3C.svg?style=flat&logo=LangChain&logoColor=white" alt="LangChain">
<img src="https://img.shields.io/badge/Docker-2496ED.svg?style=flat&logo=Docker&logoColor=white" alt="Docker">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/pandas-150458.svg?style=flat&logo=pandas&logoColor=white" alt="pandas">
<img src="https://img.shields.io/badge/Google%20Gemini-8E75B2.svg?style=flat&logo=Google-Gemini&logoColor=white" alt="Google%20Gemini">

</div>
<br>

---

## 📄 Table of Contents

- [Overview](#-overview)
- [Getting Started](#-getting-started)
    - [Prerequisites](#-prerequisites)
    - [Installation](#-installation)
    - [Usage](#-usage)
- [Features](#-features)
- [Live Demo](#-live-demo)
- [Project Structure](#-project-structure)
    - [Project Index](#-project-index)
- [License](#-license)

---

## ✨ Overview

Multi-Source-Search-with-RAG is an innovative developer tool that enables efficient querying and exploration of multiple documents and web content through AI-powered conversational interfaces. It integrates advanced NLP models, vector similarity search, and multi-format document processing to deliver accurate, real-time insights.

**Why Multi-Source-Search-with-RAG?**

This project streamlines multi-source knowledge access and enhances data interaction. The core features include:

- 🧩 **🖥️ Multi-Source Ingestion:** Upload and process PDFs, Word, Excel files, and URLs for comprehensive data coverage.
- 🚀 **🤖 Conversational AI:** Interact naturally with your data through an intuitive web interface powered by advanced language models.
- 🔍 **🎯 Embedding & Search:** Generate embeddings and perform fast similarity searches with FAISS for relevant content retrieval.
- 🐳 **🛠️ Easy Deployment:** Containerized setup with Docker and a Streamlit interface for scalable, reliable deployment.
- 🌐 **🔗 Seamless Integration:** Combines LangChain, Google Gemini Pro, and other tools for a robust, extensible knowledge platform.

---

## 📌 Features

|      | Component            | Details                                                                                     |
| :--- | :------------------- | :------------------------------------------------------------------------------------------ |
| ⚙️  | **Architecture**     | <ul><li>Modular design integrating multiple data sources</li><li>Utilizes a retrieval-augmented generation (RAG) pipeline</li><li>Separation of concerns between data ingestion, indexing, and querying</li></ul> |
| 🔩 | **Code Quality**     | <ul><li>Clear project structure with dedicated directories for modules</li><li>Consistent use of Python typing and docstrings</li><li>Uses standard libraries and well-maintained dependencies</li></ul> |
| 📄 | **Documentation**    | <ul><li>Includes a Dockerfile for containerization</li><li>README provides setup, usage, and dependency info</li><li>Comments and docstrings in code for clarity</li></ul> |
| 🔌 | **Integrations**      | <ul><li>Leverages LangChain for language model orchestration</li><li>Uses FAISS for vector similarity search</li><li>Supports multiple document formats via PyPDF2, python-docx, openpyxl</li><li>Integrates with Hugging Face models and Google Generative AI</li></ul> |
| 🧩 | **Modularity**        | <ul><li>Encapsulated components for indexing, retrieval, and query processing</li><li>Configurable via environment variables and config files</li><li>Extensible with custom data sources or models</li></ul> |
| 🧪 | **Testing**           | <ul><li>Minimal explicit testing code observed; potential for unit tests in dedicated modules</li><li>Uses tqdm for progress tracking during data processing</li></ul> |
| ⚡️  | **Performance**       | <ul><li>FAISS index (`index.faiss`) for fast similarity search</li><li>Uses sentence-transformers for embedding generation</li><li>Batch processing with tqdm to optimize throughput</li></ul> |
| 🛡️ | **Security**          | <ul><li>Uses python-dotenv for environment variable management</li><li>Potential for secure API key handling for external services</li></ul> |
| 📦 | **Dependencies**      | <ul><li>Relies on `requirements.txt` for package management</li><li>Key dependencies include langchain, faiss-cpu, sentence-transformers, pandas, requests, and streamlit</li></ul> |

---

## 🚀 Live Demo

👉 Click [here](doc-search-with-rag.streamlit.app) to check it out

---

## 📁 Project Structure

```sh
└── Multi-Source-Search-with-RAG/
    ├── Dockerfile
    ├── README.md
    ├── app.py
    ├── faiss_index
    │   ├── index.faiss
    │   └── index.pkl
    ├── img
    │   └── Architecture.jpg
    ├── rag.log
    └── requirements.txt
```

---

### 📑 Project Index

<details open>
	<summary><b><code>MULTI-SOURCE-SEARCH-WITH-RAG/</code></b></summary>
	<!-- __root__ Submodule -->
	<details>
		<summary><b>__root__</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ __root__</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/surya-adina/Multi-Source-Search-with-RAG/blob/master/Dockerfile'>Dockerfile</a></b></td>
					<td style='padding: 8px;'>- Defines the container environment for deploying the application, ensuring consistent setup of system dependencies and Python packages<br>- It facilitates streamlined deployment of the web interface built with Streamlit, enabling users to interact with the underlying data processing and AI components seamlessly<br>- This setup is essential for maintaining a reliable, portable, and scalable deployment architecture.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/surya-adina/Multi-Source-Search-with-RAG/blob/master/README.md'>README.md</a></b></td>
					<td style='padding: 8px;'>- Facilitates user interaction with multiple PDFs by enabling upload, text extraction, and conversational querying<br>- Integrates LangChain, FAISS, and Google Gemini Pro to process documents, generate embeddings, and retrieve relevant content for accurate, real-time responses<br>- Serves as the core component of a web-based AI chat application designed for seamless multi-document knowledge access.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/surya-adina/Multi-Source-Search-with-RAG/blob/master/app.py'>app.py</a></b></td>
					<td style='padding: 8px;'>- Facilitates an interactive web-based interface for uploading, processing, and indexing various document types and web content<br>- Enables users to query the integrated knowledge base through natural language, leveraging embeddings and vector similarity search<br>- Supports multi-file and URL ingestion, document chunking, and conversational AI responses, forming the core of a document-driven question-answering system within the overall architecture.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/surya-adina/Multi-Source-Search-with-RAG/blob/master/requirements.txt'>requirements.txt</a></b></td>
					<td style='padding: 8px;'>- Defines project dependencies essential for building a data processing and AI-powered document analysis platform<br>- Facilitates integration of natural language processing, document parsing, and vector similarity search, enabling seamless setup for workflows involving PDF, Word, Excel, and web content analysis, as well as advanced language model interactions within the overall architecture.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- faiss_index Submodule -->
	<details>
		<summary><b>faiss_index</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ faiss_index</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/surya-adina/Multi-Source-Search-with-RAG/blob/master/faiss_index/index.faiss'>index.faiss</a></b></td>
					<td style='padding: 8px;'>Certainly! Please provide the code file or its content, along with the project structure or additional context if available, so I can craft an accurate and succinct summary for you.</td>
				</tr>
			</table>
		</blockquote>
	</details>
</details>

---

## 🚀 Getting Started

### 📋 Prerequisites

This project requires the following dependencies:

- **Programming Language:** Python
- **Package Manager:** Pip
- **Container Runtime:** Docker

### ⚙️ Installation

Build Multi-Source-Search-with-RAG from the source and install dependencies:

1. **Clone the repository:**

    ```sh
    ❯ git clone https://github.com/surya-adina/Multi-Source-Search-with-RAG
    ```

2. **Navigate to the project directory:**

    ```sh
    ❯ cd Multi-Source-Search-with-RAG
    ```

3. **Install the dependencies:**

**Using [docker](https://www.docker.com/):**

```sh
❯ docker build -t surya-adina/Multi-Source-Search-with-RAG .
```
**Using [pip](https://pypi.org/project/pip/):**

```sh
❯ pip install -r requirements.txt
```

### 💻 Usage

Run the project with:

**Using [docker](https://www.docker.com/):**

```sh
docker run -it {image_name}
```
**Using [pip](https://pypi.org/project/pip/):**

```sh
python {entrypoint}
```

---

## 📜 License

Multi-source-search-with-rag is protected under the [LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

<div align="left"><a href="#top">⬆ Return</a></div>

---
