# 🧭 Agentic RAG Assistant for Privacy Policy Legal Queries

[**🚀 Try the Deployed App Here**](https://sigmasquad-interiit13-pathway.streamlit.app/) <!-- 🔗 sigmasquad-interiit13-pathway.streamlit.app -->

---

## 📘 Overview
We propose and implement **Agentic Retrieval-Augmented Generation (RAG)** application built to assist users in understanding and navigating **online privacy policies**.

By combining **vector database retrieval**, **web search integration**, and **graph-based reasoning**, Pathway autonomously analyzes legal information from multiple data sources — providing accurate, evidence-grounded responses for privacy-related user queries.

---

## ⚙️ Architecture

<details>
<summary>🔍 Click to expand the module breakdown</summary>

1. **🧩 Task Generator** — Breaks complex queries into subtasks using Chain-of-Thought (CoT) prompting.
2. **📚 Database Retriever (DBR)** — Retrieves contextually relevant documents from the Pathway VectorStore.
3. **🌐 Web Search Retriever (WSR)** — Gathers real-time updates from sources like Legal Stack Exchange, Reddit, and company forums.
4. **🕸 Evidence Graph Generator (EGG)** — Builds structured graph-based evidence using LlamaIndex’s SimpleGraphStore.
5. **💬 Response Generator** — Generates final responses using **Mistral-7B**, ensuring factual consistency.
6. **🛡 Guardrails** — Applies NSFW and logic consistency filters using **Guardrails AI**.
7. **⚖️ Utility Module** — (Planned) Uses **TD3-based reinforcement learning** to optimize response relevance.

</details>

---

## 🧠 Key Features
- 🧮 **Agentic reasoning pipeline** with autonomous task management.
- 🧠 **Graph-structured evidence** for grounded, interpretable outputs.
- 🔍 **Sentiment-filtered web retrieval** for higher-quality sources.
- ⚡ **Low-latency local storage** of evidence graphs.
- 🧱 **Modular design** for easy fine-tuning and scalability.

---

## 📊 Datasets & Evaluation
- **Training Dataset:** [OPP-115 Privacy Policy Dataset](https://usableprivacy.org/data) (updated URLs).
- **Testing Dataset:** Privacy Q&A Corpus (1750 annotated questions).
- **Metrics Used:**
  - ✅ **Accuracy of Correctness (AoC):** 0.91
  - ⏱ **Average Latency:** 133 s per query
  - 📖 **Knowledge F1:** Measures evidence-grounded correctness

---

## 🧩 Results Snapshot

| Pipeline Variant | AoC | Latency |
|------------------|------|---------|
| Without Evidence Graph | 0.79 | 115 s |
| Implemented Pipeline | **0.91** | **133 s** |

> Incorporating the **Evidence Graph Generator** improved accuracy by ~15% with minimal latency trade-off.

---

## 🚀 Future Work
- Integrate **multi-agent orchestration** using MetaGPT, OpenAI Swarm, or CrewAI.
- Implement **Utility Module** for reinforcement-based self-improvement.
- Experiment with **Clustered-RAG** to reduce token usage and retrieval load.
---

## 👥 Team Members
**Inter IIT Tech Meet 13.0 (High Prep Problem Statement)**

- Niranjan M
- Chandradithya J
- Adithya Ananth
- Aniket Johri
- Karthikeya M
- Sayan Kundu
- Umakant Sahu
- Deepak Yadav

---

🧭 *Developed as part of the Inter IIT Tech Meet 13.0 – High Prep Problem Statement: Pathway Project.*
