# MedicalAssistantBotAssignment


This is prototype of a medical question-answering system, created to answer user queries related to medical diseases

## Data Preprocessing

### Data Partition
[Notebook](https://github.com/brunoleme/MedicalAssistantBotAssignment/blob/main/Supportiv_DataPartition.ipynb)

As the source dataset consists in questions, with one or more reference answers, I developed a simple logic to include the last answer in the evaluation dataset and to include the previous answer to the training dataset. For the cases in which there is only one answer for a question, It was considered as a training samples.

### Vector Score Data Ingestion
[Notebook](https://github.com/brunoleme/MedicalAssistantBotAssignment/blob/main/Supportiv_VectorScore_DataIngestion.ipynb)

The proposed embedding model for the Vector Search is "sentence-transformers/all-mpnet-base-v2”, a Siamese BERT-Network fined tuned with a large dataset of 1B sentence pairs. BERT is a transformer-based model, that is very efficient to capture rich relations between words within the sentence, through its blocks with self-attention mechanism. As this "sentence-transformers/all-mpnet-base-v2” is a Siamese Network for sentence pair recognition, it is very efficient to be used to search similar documents of specific queries.

I leveraged FAISS as a vector store for this system, as a text splitter method of the chunking process, I used SentenceTransformersTokenTextSplitter with the same model as the embedding model, so the text splitting will leverage the embedding model tokenizer. The vector store was fed using only the train samples.

## System Engine
[Notebook](https://github.com/brunoleme/MedicalAssistantBotAssignment/blob/main/Supportiv_Q%26A_System.ipynb)

To implement the engine of this Q&A system, we leveraged "HuggingFaceH4/zephyr-7b-beta", that was fine-tuned model from Mistral-7B-v0.1, to act as helpful assistants.
We defined a Q&A prompt using the vector store as a retriever, with a standard RAG prompt, accessing the LLM model through a chat model interface.

It was developed a function that generates the answer, and to create guardrails, it was applied online evaluations to assess if the answer can be safely exhibit to the user. Answer will only be exhibited if it pass in the helpfulness, hallucination and document relevance evaluations. In the figure below, it can be seen the function implemented to answer the question, using the RAG System as backend.

![image](https://github.com/user-attachments/assets/3bab71ae-cfca-437b-a393-e73a61a53467)

To create an interface prototype, I leveraged a simple gradio application. Below it can be noted examples of interactio.

![image](https://github.com/user-attachments/assets/2d761cfc-8dde-4632-951d-85ab126e2fb7)

![image](https://github.com/user-attachments/assets/eda3f45e-32c0-4c32-9b3a-01c4917f0a2d)

![image](https://github.com/user-attachments/assets/ca8fb5bb-0f94-4e59-8af9-cbc1ad7473b6)


## System Evaluation

For the system online and offine evaluation, it was used LLM as a Judge, with respective prompts for the LLM evaluation.
To be completely exempt with the actual used system LLM model, it was used a distinct type of model: OpenAI "gpt-4-turbo".

### System Online Evaluation

The online evaluation is done during the answer generation, with the following metrics:
* answer_helpfulness_evaluator: measures if the answer is in fact addressing the question made.
* answer_hallucination_evaluator: measures if the system hallucinated in the answer, based on the context provided.
* document_relevance_evaluator: measures if the documents included in the context are indeed relevant to answer the question.

### System Offline Evaluation
[Notebook](https://github.com/brunoleme/MedicalAssistantBotAssignment/blob/main/Supportiv_Q%26A_SystemEvaluation.ipynb)

It was created a pipeline to assess the system performance over the evaluation dataset. The selected evaluation metrics were:
* answer_evaluator: measures if the answer are correct, in comparison with reference answer.
* answer_helpfulness_evaluator: measures if the answer is in fact addressing the question made.

Below, it can be seen the average correctness (answer_v_reference_score) and helpfulness (answer_helpfulness_score), for 200 samples from the evaluation dataset.

![image](https://github.com/user-attachments/assets/7101c32a-8e8f-4283-b586-9732e488e774)

## Potential Improvements

As improvements, we can create a feedback input to allow user to flag if the answer was useful, and rate the answer quality. Then we can create a feedback pipeline with those to help the system to focus on useful and top rating references only. Those feedbacks can be used also to fine-tune the LLM model used in the system. Furthermore, it can be included fallback models, in case of the main model were unable to respond the question.
