# MedicalAssistantBotAssignment
Medical  Assistant Bot Assignment


This is prototype of a medical question-answering system, created to answer user queries related to medical diseases. This project was developed following the steps below.

## Data Preprocessing

### Data Partition

As the source dataset consists in questions, with one or more reference answers, I developed a simple logic to include the last answer in the evaluation dataset and to include the previous answer to the training dataset. For the cases in which there is only one answer for a question, I considered it as a training sample.

### Vector Score Data Ingestion

I’m proposing the usage of the model "sentence-transformers/all-mpnet-base-v2”, a Siamese BERT-Network fined tuned with a large dataset of 1B sentence pairs. BERT is a transformer-based model, that is very efficient to capture rich relations between words within the sentence, through its blocks with self-attention mechanism. As this "sentence-transformers/all-mpnet-base-v2” is a Siamese Network for sentence pair recognition, it is very efficient to be used to search similar documents of specific queries.

I leveraged FAISS as a vector store for this system, as a text splitter method of the chunking process, I used SentenceTransformersTokenTextSplitter with the same model as the embedding model, so the text splitting will leverage the embedding model tokenizer.

The vector store was feed using only the train samples.

## System Engine

To implement the engine of this Q&A system, we leveraged "HuggingFaceH4/zephyr-7b-beta" as LLM model, that was fine-tuned from Mistral-7B-v0.1 to act as helpful assistants.
We defined a Q&A prompt using the vector store as a retriever, with a standard RAG prompt, accessing the LLM model through a chat model interface.

As guardrails, we created a function that generates the answer, and then apply online evaluations as guardrails to assess if the answer can be safely exhibit to the user. Answer will only exhibit to the user if it pass in the helpfulness, hallucination and document relevance evaluations.

To create an interface prototype, I leveraged a simple gradio application.
Examples of interaction.

## System Evaluation

For the system online and offine evaluation, it was used LLM as a Judge, with respective prompts for the LLM evaluation.
To be completely exempt with the actual used system LLM model, we used OpenAI "gpt-4-turbo" model.

### System Online Evaluation

For online evaluation, we chose the following metrics:
* answer_helpfulness_evaluator: measures if the answer is in fact addressing the question made.
* answer_hallucination_evaluator: measures if the system hallucinated in the answer, based on the context provided.
* document_relevance_evaluator: measures if the documents included in the context are indeed relevant to answer the question.

### System Offline Evaluation

I created a pipeline to assess the system performance over the evaluation dataset. The selected evaluation metrics are:
* answer_evaluator: measures if the answer are correct, in comparison with reference answer.
* answer_helpfulness_evaluator: measures if the answer is in fact addressing the question made.

Results Table

## Potential Improvements

As improvements, we can create a feedback input to allow user to flag if the answer was useful, and rate the answer quality. Then we can create a feedback pipeline with those to help the system to focus on useful and top rating references only. Those feedbacks can be used also to fine-tune the LLM model used in the system.
