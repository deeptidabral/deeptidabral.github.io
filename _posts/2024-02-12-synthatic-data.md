---
title: 'Overview of synthatic data'
date: 2024-02-12
permalink: /posts/2024/02/synthatic-data/
toc: false
---

# Overview of Synthetic Data

## 1.1 What is synthetic data?

Synthetic data is artificial but realistic data generated to increase the availability of data required for critical stages of the AI/ML model lifecycle where the use case might not have sufficient data samples. It is critical to note that even though synthetic data is completely fake, it is representative of real-world data, making it usable for AI model training, testing, supervised fine-tuning, and governance purposes. Synthetic data generation (SDG) creates new data points/samples mimicking the real-world data variables and data samples.

## 1.2 When is it needed?

Synthetic data is needed when actual data is either or all the following:

- **Limited (in availability)** – Real data can be scarce or difficult to source. Synthetic data can be generated to simulate uncommon or rare events, edge cases, and complex scenarios to ensure meaningful modeling efforts.
- **Sensitive and confidential** – Some AI use cases, such as those in the healthcare industry, may have access to sensitive data fields identifying patients (PII) and their medical history. Synthetic data does not suffer from privacy concerns.
- **Biased** – Existing data can be biased and imbalanced, thus not representative of various population groups (e.g., customer segments, transaction categories, business types, etc.) or subgroups. If understanding the behavior of these groups or subgroups is critical to the AI application, then real data would not be sufficient to ensure fairness in AI models built using it.
- **Expensive** – Collecting real data (e.g., via surveys) can be expensive, hugely manual, and time-consuming, yet fall short of the target data volume needed for all efforts. On the other hand, synthetic data can generate data at scale and be used across the AI product lifecycle across development, testing, production, and maintenance.

![My Image](/assets/images/Lifecycle.png)

## 1.3 What are the leading benefits of synthetic data?

Good-quality synthetic data has several applications, such as the following:

- **Provisioning of data-at-scale**: Provides realistic and safe data at scale for training, testing, and validating AI models and applications during development. As concept drifts emerge in data while AI models are in production, synthetic data can be generated for evolving data distributions to ensure ongoing testing and retraining of models.
- **Handling bias in data**: Real-world data may suffer from biases and inaccuracies. Synthetic data, curated to introduce diversity, removes bias from existing datasets and enables extensive testing of a wide variety of potential vulnerabilities and points of failure in AI models/applications.
- **Scalable and seamless model testing**: Specialized testing of AI models, focused on specific business objectives, can be done. To ensure sufficient adversarial testing of AI models, synthetic data can be useful if original datasets lack edge case scenarios.
- **Adherence to data privacy and regulations**: Since synthetic data is fake, the newly generated data is never non-compliant with data protection regulations such as GDPR (EU), CCPA (US), HIPAA (US), PIPEDA (Canada), and more.
- **Maintaining data privacy**: Synthetic data records created are fully anonymous and do not preserve any direct mapping or linkages to real business entities (e.g., customers, orders, transactions, etc.). It can be seamlessly shared across user roles and teams for collaboration without significant concerns about data safety.

![My Image](/assets/images/Privacy.png)

## 1.4 Types of synthetic data

Synthetic data can be broadly categorized across two key dimensions: its **generation technique** and its **modality**.

### (a) Generation technique

| Synthetic Data Type | Description | Challenges |
|---------------------|-------------|------------|
| **Mock data** | Generated based on a rules engine with predefined rules, constraints, and parameters (e.g., generate mock data for rules such as ‘loyal’ shoppers of a retailer purchasing at least _n_ times and spending > $X in a week; ‘high risk’ customer segment of a credit card provider with >_n_ open credit lines and frequent defaults in the past). | May resemble real data in structure but may not represent rich patterns (e.g., volatility, multi-dimensional relationships, rare events). Rules can become complex, making compliance with business constraints difficult to implement. |
| **AI-generated synthetic data** | Generated algorithmically using existing data samples as seed. Created by deep generative algorithms such as GPTs, GANs, VAEs, and more. | Dependent on availability of large enough samples to seed the data generation. Risk of overfitting, making it a biased and unrepresentative sample. Determining optimal synthetic data generators for specific datasets/use cases. Balancing bias-variance tradeoff. |

### (b) Modality

| Synthetic Data Type | Description | Challenges |
|---------------------|-------------|------------|
| **Structured synthetic data** | Fictitious, life-like data that mimics the structural characteristics (statistical distributions) and properties of real data. Created using statistical modeling to simulate data that closely resembles the original data in terms of its statistical properties and relationships between variables. | High probability of error propagation and bias amplification if actual underlying data has inherent biases and unidentified errors. |
| **Unstructured synthetic data** | Fake data in free-form modalities such as images, videos, text, and audio. Created by deep generative algorithms such as LLMs, Diffusion models, GANs, and more. | May lack subtle details and relationships across data variables and samples present in real data. Sophisticated validation of the synthetic data can be quite complex and require deep expertise in unstructured data. |

## 1.5 What are the potential challenges or pitfalls related to synthetic data?

- **Risk of model collapse / curse of recursion** – Iteratively training models on data generated by them can degrade model quality due to the loss of original data distributions. However, this can be managed through human expertise and careful data generation efforts.
- **Evaluation strategy and metrics** – Development and deployment of every use case require careful and customized evaluation of synthetic data to ensure accurate real-world data representation and meaningful model results. Validation of synthetic data can be complex and time-consuming.
- **Bias amplification** – Undetected or unfixed bias in real data can perpetuate bias in newly generated synthetic data, impacting model performance, insights, and business decisions.
- **Repeated need for regeneration** – With emerging data drifts and concept drifts, synthetic data must be regenerated periodically to adequately mimic evolving data distributions.
- **Data quality issues**:
  - **Correctness and diversity**: Synthetic data should mimic input data while introducing diversity and eliminating bias for nuanced AI modeling. If the dataset is not varied enough, the model will not generate appropriate responses for rare events or underrepresented groups.
  - **Bias inheritance**: SDG can inadvertently propagate inaccuracies or biases present in their pre-training data, leading to outputs that may not be factual.
  - **Presence of hallucinations**: Synthetic data generated by LLMs can be inaccurate or fictitious due to the quality of the training data, especially if it contains inaccuracies and errors.

It is important to understand the difference between synthetic data generation and data anonymization/data masking.


# 2. Generative AI Techniques for SDG

## 2.1 How can GenAI be used for synthetic data generation?

Generative AI can be used for synthetic data generation by leveraging deep learning models such as GANs, VAEs, Diffusion Models, and LLMs to create realistic structured (tabular), unstructured (text, images, audio), and multimodal datasets, facilitating privacy-preserving analytics, AI model training, and data augmentation across diverse domains.

| S. No. | Data Modality        | GenAI Models / Algorithms | Illustrative Data Generated | Popular Use Cases | Potential Challenges / Limitations |
|--------|----------------------|--------------------------|----------------------------|-------------------|-----------------------------------|
| 1      | **Tabular Data (Structured Data)** | Bayesian Networks, GANs (e.g., CTGAN, Tabular GAN), TimeGAN for sequential data, Autoencoders for anomaly detection | Financial, healthcare, or business datasets | Customer behavior (account balance, product holding) | Maintaining relational integrity rules of actual data, Privacy concerns (re-identification risks), Bias in generated datasets |
| 2      | **Text** | Prompt engineering with LLMs (e.g., GPT, BERT, T5, LLaMA), Retrieval-Augmented Generation (RAG), Knowledge Graphs for text consistency | Customer conversation and feedback data | Chatbot training, sentiment analysis, and text summarization, fake news detection | Hallucination issues in LLMs, Ensuring fact-checking and logical consistency, Bias in generated text |
| 3      | **Image (CV)** | Latent space interpolation in GANs, CLIP-guided image synthesis, Style transfer for dataset expansion; GANs (e.g., StyleGAN, BigGAN) and Diffusion Models (e.g., Stable Diffusion) | Human faces, medical images (MRI, X-rays), product images for e-commerce, rare object detection datasets | Facial recognition, medical imaging (anonymization & augmentation), self-driving car training, deepfake detection | Artifacts in GAN-based images, Ethical concerns (deepfakes), Ensuring medical image fidelity |
| 4      | **Audio & Speech** | Text-to-Speech (TTS) and Voice Cloning models (e.g., WaveNet) | Speech samples, language accents, emotion-augmented speech for virtual assistants | Automatic speech recognition (ASR), voice assistants, AI-generated audiobooks, voice cloning | Synthetic speech sounding unnatural, Difficulty in capturing emotions accurately |
| 5      | **Time-series Data** | Recurrent Neural Networks (RNNs), Transformers, and GANs | Financial market data, sensor readings, and IoT device logs | Commodity price forecasting, algorithmic trading | Ensuring time-series consistency, Handling concept drift in long-term data |

# 3. Benefits of Synthetic Data for Agentic AI

## 3.1 How can synthetic data be used to augment agentic AI?

| S. No. | Use Case | Description | Example | GenAI Algorithms | Challenges & Limitations | Mitigation Strategies |
|--------|----------|-------------|---------|------------------|--------------------------|-----------------------|
| 1 | **Simulating User Interactions for Fine-Tuning** | Improve training by introducing diversity in simulated conversations, user queries, and behavioral patterns. | Simulating customer-agent conversations to resolve customer concerns and consultation responses. | LLMs (e.g., GPT, Claude, LLaMA), Retrieval-Augmented Generation (RAG), Few-shot prompting | Risk of model overfitting to synthetic (fake) interactions, Ensuring realism in generated conversations. | Using human-in-the-loop (HITL) validation, Implementing hybrid strategy for model training with real and synthetic dialogues. |
| 2 | **Enhancing Multimodal Agents** | Synthetic multimodal (text, image, audio) data enables AI agents to understand and process multi-sensory inputs. | Generating synthetic voice commands for a voice assistant to improve speech recognition in noisy environments. | CLIP for text-image understanding, Speech2Text models (e.g., Whisper, Wav2Vec), Multimodal transformers | Difficulties in aligning synthetic multimodal data with real-world user interactions. | Fine-tuning multimodal agents with real-world sensor data and human feedback. |
| 3 | **Training Agents for Task Generalization** | Synthetic workflows can help fine-tune task execution and problem-solving capabilities across different contexts. | Creating synthetic task descriptions to train AI agents in handling rare or complex scenarios (e.g., legal contract drafting). | Self-supervised learning, Reinforcement Learning from Human Feedback (RLHF) | Ensuring synthetic workflows align with actual real-world task expectations. | Calibrating synthetic data distributions to match real-world datasets, Continuous domain adaptation. |
| 4 | **Error Handling and Recovery Strategies** | Synthetic failure scenarios help agents learn self-correction, automated recovery, and fallback strategies. | Simulating API failures for an autonomous trading bot to train it on recovery and exception-handling strategies. | Fine-tuned LLMs with synthetic failure dialogues, Policy gradient-based reinforcement learning | Balancing failure simulations with real-world unpredictability, Avoiding false positives in error detection. | Using adversarial training to expose AI agents to a variety of failure scenarios. |
| 5 | **Reinforcement Learning for AI Agents** | Synthetic rewards, environments, or policies in reinforcement learning enable scalable AI training for decision-making. | Training an AI agent to optimize personalized readings based on simulated customer behavioral data and feedback. | Deep Q Networks (DQN), PPO-based reinforcement learning, Curriculum learning for stepwise training | Overfitting to synthetic environments, Risk of unrealistic reward functions leading to suboptimal policies. | Combining synthetic training with real-world reinforcement environments, Reward function optimization. |
| 6 | **Bias Reduction and Fairness Training** | Generating diverse synthetic user profiles and interactions to mitigate bias in AI decision-making and recommendations. | Generating synthetic demographic data to train AI hiring bots to avoid biases in job recommendations. | Counterfactual data generation using GANs, Debiasing techniques in synthetic dataset curation | Synthetic demographic data might not fully generalize to real-world diversity, Potential bias in data generation models. | Leveraging fairness-aware synthetic data curation methods, Regular audits for synthetic dataset biases. |
| 7 | **Testing AI Agent Robustness and Security** | Using adversarial synthetic data to test AI agents against edge cases, security threats, and adversarial attacks. | Introducing adversarial prompts to test AI chatbots for prompt injection attacks and misinformation resilience. | Adversarial AI techniques (e.g., adversarial perturbations, red-teaming AI), AI security stress testing frameworks | Ensuring adversarial synthetic data does not create unintended vulnerabilities, Maintaining ethical AI boundaries. | Deploying adversarial robustness testing, Implementing real-world scenario benchmarking for AI resilience. |

---

## 3.2 What are the key impact levers of synthetic data in the evaluation of agents?

Synthetic data can enable agent evaluation in a robust and comprehensive way.

| S. No. | Dimension of Evaluation | Description (What will synthetic datasets test?) | What is being evaluated in agents? | Why it matters? | Illustrative Test Scenarios (Customer Service Chatbot in Retail Banking) |
|--------|------------------------|-----------------------------------|------------------------------|-----------------|------------------------------------------------|
| 1 | **Complex Problem Solving and Logical Reasoning** | Test for logical capabilities, multi-step reasoning tasks, deductive, and inductive reasoning tasks. | Is the agent capable of performing step-by-step reasoning rather than simple pattern matching? Multi-step or abstract reasoning tasks? | Ensures agent is performing robust step-by-step reasoning. Identifies weaknesses in multi-step or abstract reasoning tasks. | Testing duplicate transaction disputes requiring the agent to look up and identify multiple identical charges, analyze account histories, verify merchant details. |
| 2 | **Fact-checking and Correctness** | Test for fake citations, claims, or facts inserted by synthetic data in responses. Measure overall reliability and accuracy. | Is the agent capable of retrieving high-relevance, accurate documents for a specific RAG system? Detecting fake claims or misinformation in synthetic datasets? | Reduces the risk of hallucination. Ensures grounded responses based on facts. | Testing chatbot for queries with well-known correct responses such as loan eligibility, credit limit increase, product-related explanations, and fee calculations. |
| 3 | **Alignment with Human Intent (Ethics, Fairness, Bias Mitigation)** | Test for toxic, biased, unethical responses using adversarial prompting. Measure responses across different cultural and socioeconomic populations. | Is the agent providing ethical, fair, and non-toxic responses to adversarial prompts? Ensuring compliance with ethical and regulatory standards? | Detects biases and toxicity against specific marginalized user groups. Ensures compliance with responsible AI principles. | Testing chatbot for consistency in service quality across income groups, demographics, and ethnic groups. |
| 4 | **Usage of Toolset by Agents** | Test if agents correctly use tools by generating datasets specifying actions requiring specific tool invocations. | Is the agent invoking the right tools for specific actions? E.g., ArvixQueryRun for research papers or Tavily Search API for online searches. | Tests decision-making flow of the agent to invoke the right tool for the job. Detects failure points when tools return unexpected results. | Testing chatbot’s ability to use core banking access (transaction history, product holding), fraud detection tools, human agent handoff tool, etc. |
| 5 | **Data Drift Monitoring** | Test for alerts/thresholds on model risk parameters, anomaly detection, and major data distribution changes. | Is the agent handling edge cases? Detecting data distribution shifts and model resilience against them? | Detects changing population patterns and updates models for adaptation. Manages concept drift. | Using time-series synthetic data to test for changes in customer behavior due to external factors (e.g., competition, economic shifts). |
| 6 | **Real-time Performance (Latency)** | Test response time under high-volume, complex user queries. | Is the agent capable of handling high query volumes per second? Managing failures when APIs are inaccessible? | Stress-tests agent for real-life edge scenarios with production constraints. | Using synthetic data to simulate high load requirements, resource-intensive queries, and complex conversations. |
| 7 | **Safety, Adversarial Testing & Jailbreak Prevention** | Test resilience against exploits such as malicious prompts. | Is the agent capable of identifying adversarial prompts (e.g., scam messages, phishing attempts)? Preventing PII leakage? | Ensures agent responses do not contain harmful or toxic results. Improves handling of adversarial attacks. | Using synthetic data to simulate scenarios like identity theft, social engineering attacks, and fraudulent transaction attempts. |

---

## 3.3 What are some effective strategies to leverage synthetic data to improve Agentic AI applications?

### **Model Training**
- **Data Augmentation**: Expands training datasets with diverse synthetic samples, simulating edge cases and rare events.
- **Few-Shot Learning**: Provides high-quality synthetic examples for fine-tuning.
- **Bias Reduction**: Balances underrepresented scenarios in training data.
- **Privacy-Preserving Training**: Avoids reliance on sensitive real-world data.

### **Model Testing**
- **Robustness Testing**: Evaluates models under adversarial or unseen conditions.
- **Edge Case Generation**: Creates rare or extreme scenarios to assess failure modes.
- **Performance Benchmarking**: Standardizes evaluation across different test conditions.
- **Simulated User Behavior**: Tests AI agents in synthetic real-world interactions.

### **Supervised Fine-Tuning**
### **Model Governance**


---

## 4. Evaluation and Validation of Synthetic Data

### 4.1 What should be the leading tenets of evaluating synthetic data?
- **Perceptual Assessment** –  This means evaluating how well synthetic text, images, audio, or video align with human perception and real-world semantics. Comparing statistical properties of real and synthetic data and determining if they follow same distribution patterns would help understand how far or close is synthetic data to real data. Some examples of such comparisons could be:
  - Similarity between word frequency distributions of real-world chats of customer and customer service chatbots and synthetic samples created using them
  - Similarity between color, and spatial distributions of real-world geo-spatial data and synthetic projections of the same
  - Similarity between pitch and tone distributions of audio data of customer feedback and synthetic samples created from it
- **Semantic Consistency Checking** – This includes checking whether synthetic data maintains logical coherence, retains meaningful relationships present in underlying data, and demonstrates contextual accuracy across its content. Synthetic data should be coherent across modalities such as text-image / caption-content alignment.
- **Task-Specific Performance Evaluation & Model Generalization** – This includes assessing the performance and effectiveness of synthetic data in improving AI model performance. Comparing performance of models trained on synthetic data versus models trained on real data can help understand value of synthetic data. Model performance metrics would vary based on the nature of use case.
- **Bias, Authenticity & Ethical Risks** –  This includes detecting potential sources of bias such as representation bias and contextual bias, finding hallucinations in synthetic text, and determining misinformation in synthetic datasets. 

### 4.2 What are some leading techniques of validating synthetic data?
- Techniques such as KL Divergence, cosine similarity of text embeddings can help determine similarity between real and synthetic distributions. Metrics such as BLEU score, and ROUGE score help benchmark coherence and logical consistency of synthetic text. Additionally, adversarial robustness testing can help detect weaknesses in synthetic data.

- Validation methods should check for meaningful and logically consistent semantic properties for its intended applications. Contextual coherence and logical flow analysis using coherence scoring techniques, semantic similarity and embedding validation using vector space alignment and similarity of embeddings, and fact checking using retrieval-based fact verification systems and fact-checking APIs are some of the leading methods.

- For assessing task-specific performance using synthetic data, it is crucial to analyze use case specific performance metrics such as the following:
  - NLP applications: text summarization, similarity metrics of embedding models, context precision, and context recall
  - Computer vision applications: object detection metrics, image comprehension quality, facial recognition reliability 

- Metrics for assessing bias such as diversity metrics (to determine representation of underrepresented groups), fairness audits by using bias detection tools and adversarial testing can help detect and remove bias from synthetic data. 

As an example from an implementation perspective, Synthetic Data Metrics (SDMetrics) is a Python library for evaluating tabular synthetic data. This can be used for generating, evaluating, and improving synthetic data. SDMetrics is a suite of validation tools to assess the quality of synthetic data based on statistical similarity, fidelity, and utility compared to real-world datasets. This package includes modules to conduct distributional similarity tests, and privacy risk evaluations. It also supports custom metric creation, allowing domain-specific validations such as bias detection and fairness analysis. Evaluating unstructured synthetic data requires specialized perceptual and semantic validation techniques like CLIP for images or BERT embeddings for text.

## 5. SDG and Using Synthetic Data with an AI Agent - Notebook

This notebook uses the `ragas` library and accomplishes the following key tasks:
- Use RAGAS to Generate Synthetic Data
Ragas is a Python library designed for generating and evaluating synthetic data specifically for Retrieval Augmented Generation (RAG) systems. 
Core functionality is to leverage knowledge graph-based algorithms to generate diverse synthetic question-answer pairs. The library explores documents and connects related content, enabling the creation of various question types and complexities.
It supports two main categories of questions:
Single Hop: Straightforward questions requiring information from a single source
Multi Hop: Complex questions needing multiple steps of reasoning and information from multiple documents
The library provides a TestsetGenerator class that handles synthetic data creation. Users can define the distribution of query types using built-in synthesizers.
- Another cool feature is dashboard visualization through app.ragas.io
- Load synthetic data into a LangSmith Dataset to implement tracing
- Evaluate a basic RAG pipeline against the synthetic test data
- Make changes to this pipeline
- Evaluate the modified pipeline

Ragas employs an evolutionary generation paradigm to create diverse question types, including
- Reasoning questions
- Conditional queries
- Multi-context scenarios

(1) **qa_evaluator:** This QA (Question-Answer) evaluator checks how the generated response (prediction) aligns with an expected or reference answer.
- The evaluator argument is set to "qa" suggesting this evaluator will leverage a built-in evaluator meant for general QA tasks.
- The LLM (eval_llm) assesses the quality of the response

(2) **labeled_helpfulness_evaluator:** This evaluator checks how helpful the response is.

- It uses a custom labeled criteria ("labeled_criteria") with a specific focus on helpfulness.
- The prepare_data function structures the input for evaluation: prediction (model's generated response), reference (expected answer), input (query)
- The LLM (eval_llm) dteremines how helpful the generated response is, benchmarked in comparison to the reference.

(3) **dope_or_nope_evaluator:** This evaluator is assessing "dopeness" (coolness).
- It uses a custom "criteria" evaluator with the criterion "dopeness", checking if the response is dope / lit / cool.
- The LLM (eval_llm) determines whether the generated response meets this informal, subjective standard.


## GitHub Repo: 
[`https://github.com/deeptidabral/AIE5/blob/main/07_Synthetic_Data_Generation_and_LangSmith/Synthetic_Data_Generation_RAGAS_%26_LangSmith_Assignment.ipynb`](https://github.com/deeptidabral/AIE5/blob/main/07_Synthetic_Data_Generation_and_LangSmith/Synthetic_Data_Generation_RAGAS_%26_LangSmith_Assignment.ipynb)

![My Image](/assets/images/Turing.png)

## APPENDIX: 
## References

- [https://arxiv.org/pdf/2403.04190](https://arxiv.org/pdf/2403.04190)
- [https://aws.amazon.com/what-is/synthetic-data/](https://aws.amazon.com/what-is/synthetic-data/)
- [https://arxiv.org/pdf/2409.12437](https://arxiv.org/pdf/2409.12437)
- [https://mostly.ai/what-is-synthetic-data](https://mostly.ai/what-is-synthetic-data)
- [https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary](https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary)
- [https://arxiv.org/html/2402.07712](https://arxiv.org/html/2402.07712)
- [https://docs.ragas.io/en/stable/](https://docs.ragas.io/en/stable/)
- [https://www.moveworks.com/us/en/resources/blog/synthetic-data-for-ai-development](https://www.moveworks.com/us/en/resources/blog/synthetic-data-for-ai-development)
- [https://docs.ragas.io/en/v0.1.21/concepts/testset_generation.html](https://docs.ragas.io/en/v0.1.21/concepts/testset_generation.html)
- [https://docs.ragas.io/en/latest/getstarted/rag_testset_generation/#knowledgegraph-creation](https://docs.ragas.io/en/latest/getstarted/rag_testset_generation/#knowledgegraph-creation)

