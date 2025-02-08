---
layout: post
title: "Leveraging Synthetic Data Generation (SDG) for evaluating Agentic AI"
date: 2025-02-08
categories: [Agentic AI eval]
tags: [GitHub Pages, Jekyll, Blogging]
author: "Deepti Dabral"
excerpt: "This is a short summary of the blog post."
permalink: /leveraging-sdg-for-agentic-ai/
---

Needed when < actual data may be sensitive, confidential, or limited in availability>
While there are several techniques of SDG, this post is focused on using GenAI for SDG.

Structured data: Synthetic data refers to artificially generated data that mimics the characteristics and patterns of real-world data. It is created using statistical models or algorithms to simulate data that closely resembles the original data in terms of its statistical properties, distribution, and relationships between variables.
Unstructured data: 
LLMs, such as ChatGPT, have revolutionized our approach to understanding and generating human-like text, providing a mechanism to create rich, contextually relevant synthetic data on an unprecedented scale. This synergy is pivotal in addressing data scarcity and privacy concerns, particularly in domains where real data is either limited or sensitive. By generating text that closely mirrors human language, LLMs facilitate the creation of robust, varied datasets necessary for training and refining AI models across various applications.

XXXXX  

XXXXX

# 2. Using Generative AI for SDG #

XXXXX  

XXXXX

| S. No. | Data modality                  | GenAI models / algorithms                                               | Egs of data categories generated                                                                      | Popular use cases                                                                                                                                                           |
|--------|--------------------------------|-------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1      | Tabular data (structured data) | GANs (e.g., CTGAN, Tabular GAN)                                         | -financial data (stock prices, P&L projections) -customer behavior (account balance, product holding) | Marketing analytics use cases -Up-sell / Cross-sell -Hyperpersonalization-at-scale -NPTB / recommender systems -Churn prediction, Credit risk underwriting Collections Risk |
| 2      | Text                           | LLMs (e.g., GPT, BERT, T5, LLaMA)                                       | Customer conversation and feedback data                                                               | -Conversational AI (chatbots) -Sentiment analysis -Theme extraction -Summarization                                                                                          |
| 3      | Image                          | GANs (e.g., StyleGAN, etc) Diffusion Models (e.g., Stable Diffusion)    | Images                                                                                                | -Facial recognition -Medical imaging -Autonomous driving                                                                                                                    |
| 4      | Audio & Speech                 | Text-to-Speech (TTS) and Voice Cloning models (e.g., Tacotron, WaveNet) | Speech samples                                                                                        | -Automatic speech recognition -Virtual assistants                                                                                                                           |
| 5      | Time-series data               | Recurrent Neural Networks (RNNs), Transformers, and GANs                | -Financial market data -Sensor readings -IoT device logs                                              | -Commodity price forecasting -Algorithmic trading                                                                                                                           |

XXXXX  

XXXXX


# 3. Using SDG to improve Agentic AI #

XXXXX  

XXXXX

|     S. No.    |     Use Case                                        |     Description                                                                                     |     Example                                                                             |
|---------------|-----------------------------------------------------|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
|     1         |     Simulating User Interactions for Fine-Tuning    |     Improve training by introducing diversity in   synthetic datasets                               |     Simulating astrology client-agent   conversations with varying emotional tones      |
|     2         |     Enhancing Multimodal Agents                     |     Synthetic multimodal (text, image, audio)   data improves agents that process diverse inputs    |     Generating synthetic voice commands for a   voice-enabled AI astrologer             |
|     3         |     Training Agents for Task Generalization         |     Synthetic workflows can help fine-tune task   execution agents.                                 |     Creating synthetic task descriptions to   train an autonomous research assistant    |
|     4         |     Error Handling and Recovery Strategies          |     Synthetic failure scenarios help agents   learn from mistakes.                                  |     Simulating API failures for an autonomous   trading bot                             |
|     5         |     Reinforcement Learning for AI Agents            |     Synthetic rewards, environments, or policies   improve reinforcement learning models            |     Training an AI astrologer to optimize   personalized recommendations                |

XXXXX  

XXXXX

# 4. Key benefits of Synthetic Data in Agentic AI evaluation #

XXXXX  

XXXXX

| S. No. | Dimension of evaluation                                         | Description (What will synthetic datasets generated will test?)                                                                                                                                                                     | What is being evaluated in agents?                                                                                                                                                                                                                                                                                                                                                                   | Why it matters? (Impact of evaluation)                                                                                                                                           |
|:------:|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|    1   | Complex problem solving and Logical reasoning                   | Synthetic datasets will test for:  - logical capabilities  - multi-step reasoning tasks  - deductive, inductive reasoning tasks                                                                                                     | Is the agent capable of performing:  - Step-by-step reasoning rather than simple pattern matching?  - Multi-step or abstract reasoning tasks?                                                                                                                                                                                                                                                        | Ensures agent is performing robust step-by-step reasoning   Identifies weaknesses in implementation of multi-step or abstract reasoning tasks.                                   |
|    2   | Fact-checking and Correctness                                   | Synthetic datasets will test for:  - Any fake citations / claims / facts, inserted by synthetic data, in responses  - Overall reliability of responses indicated by the accuracy of responses generated                             | Is the agent capable of:  - Retrieval of high-relevance and accurate documents for a specific RAG system?  - Detecting fake claims or misinformation in the synthetic datasets?                                                                                                                                                                                                                      | Reduces risk of hallucination  Ensures grounded responses based on actual facts                                                                                                  |
|    3   | Alignment with human intent (ethics, fairness, bias mitigation) | Synthetic data will test for:  - Toxic, biased, unethical responses by adversarial prompting  - Responses for user queries from different cultural /ethnic, and socioeconomic populations                                           | Is the agent capable of providing:  - ethical, fair, and non-toxic responses to adversarial (biased, misleading) prompts, thus, ensuring compliance with ethical and regulatory standards (and other core tenets of responsible AI)?                                                                                                                                                                 | Detects biases and toxicity against specific marginalized user groups  Ensures compliance with responsible AI principles.                                                        |
|    4   | Usage of toolset by agents                                      | Synthetic data will test for:  - if the agents in the system are using the correct tool for specific action(s) by generating datasets specifying search or research actions to invoke specific nodes / edges in an agentic workflow | Is the agent capable of:  - Invoking the right tools from its toolset for specific actions? E.g., is it invoking ArvixQueryRun when it must analyze research papers or invoking Tavily Search API when it must conduct an online search  E.g. For a specific example, assume agents are equipped with a variety of tools (e.g. ArvixQueryRun, Tavily’s Search API from LangChain community of tools) | Tests decision making flow of the agent to invoke right tool for the job  Detects failure points when tools return unexpected, unmanaged results                                 |
|    5   | Data drift monitoring                                           | Synthetic data will test for:  - Alerts / thresholds of model risk’s parameters  - Anomaly detection (erratic / abrupt changes in data values across key variables) strategy  - Major changes in data distributions                 | Is the agent capable of:  - handling the edge cases?  - Detecting data distribution shifts and model resilience against them?                                                                                                                                                                                                                                                                        | Detects changing population patterns and updates / tunes model for adaptation  Manages concept drift (model performance degrades due to evolving data distributions) proactively |
|    6   | Real-time performance                                           | Synthetic data will test for:  - Average response time of agent when inundated with high-volume, complex user queries  - Latency across different complex situations                                                                | Is the agent capable of:  - Handling XX volume of queries per second (specific latency / response times)?  - Handling failures of any nodes / edges (e.g. data inaccessible due to API failures)?                                                                                                                                                                                                    | Performs stress-testing for real-life edge scenarios requiring constraints applicable in production environment                                                                  |
|    7   | Safety, Adversarial Testing & Jailbreak Prevention              | Synthetic data will test for:  - Resilience of agentic AI system in face of exploits such as malicious prompts                                                                                                                      | Is the agent capable of:  - Identifying adversarial prompts such as scam messages, bot-generated messages, cybersecurity attacks, phishing, identity thefts, jailbreaking attempts /?  - Preventing leakage of private information (such as PII data of customers)?                                                                                                                                  | Ensures agent responses do not contain harmful, and toxic results and mature handling of adversarial attacks                                                                     |

XXXXX  

XXXXX


# 5. Using Synthetic data for training Agentic AI #


# 6. Validating and evaluating Synthetic data #


# 7. LLMOps practices to ensure building production-grade deployments #


# 8. Challenges / Pitfalls of Synthetic data #


# References #

1. <https://arxiv.org/pdf/2403.04190>
2. <https://aws.amazon.com/what-is/synthetic-data/>
3. <https://arxiv.org/pdf/2409.12437>
4. <https://docs.ragas.io/en/stable/concepts/>

