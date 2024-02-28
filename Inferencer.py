from haystack import Pipeline
from haystack.utils import Secret
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
from haystack.components.readers import ExtractiveReader
from haystack.components.generators import GPTGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.generators import OpenAIGenerator
from dataloader import DataLoader


chroma_store_loader = DataLoader()
class Inferncer:
    
    def __init__(self):
        self.chroma_store = chroma_store_loader.chroma_store

    def OpenAI(self,query):
        template = """
        Answer all the questions in the following format and based on Aditya.
        and if the answer is not present in the context, then generate the answer using the LLM model.
        Context:
        {% for doc in documents %}
        {{ doc.content }}
        {% endfor %}
        Question: {{question}}
        Answer:
        """

        prompt_builder = PromptBuilder(template=template)
        retriever = ChromaQueryTextRetriever(document_store = self.chroma_store)
        #ExtractiveReader to extract answers from the relevant context
        api_key = Secret.from_token("API-KEY")
        llm = OpenAIGenerator(model="gpt-3.5-turbo-0125",api_key=api_key)
        # reader = ExtractiveReader(model="deepset/roberta-base-squad2-distilled")

        extractive_qa_pipeline = Pipeline()
        extractive_qa_pipeline.add_component("retriever", retriever)
        # extractive_qa_pipeline.add_component("reader",reader)
        extractive_qa_pipeline.add_component(instance=prompt_builder,   name="prompt_builder")
        extractive_qa_pipeline.add_component("llm", llm)

        # extractive_qa_pipeline.connect("retriever.documents", "reader.documents")
        extractive_qa_pipeline.connect("retriever.documents", "prompt_builder.documents")
        extractive_qa_pipeline.connect("prompt_builder", "llm")


        
        # Define the input data for the pipeline components
        input_data = {
            "retriever": {"query": query, "top_k": 1},
            "prompt_builder": {"question": query},
            # "reader": {"query": query}
            # Use 'max_tokens' instead of 'max_new_tokens'
        }

        # Run the pipeline with the updated input data
        results = extractive_qa_pipeline.run(input_data)
        return results
    
    def LlamaCpp(self,query):
        template = """
    `    Answer all the questions in the following format and based on Aditya 
        and if not found generate answer accordingly using the given information.

        Context:
        {% for doc in documents %}
        {{ doc.content }}
        {% endfor %}
        Question: {{question}}
        Answer:
        """

        prompt_builder = PromptBuilder(template=template)
        # retriever = ChromaQueryTextRetriever(document_store = chroma_store)
        #ExtractiveReader to extract answers from the relevant context

        llm = LlamaCppGenerator(
        model_path="openchat-3.5-1210.Q3_K_S.ggml", 
        n_ctx=10000,
        n_batch=256,
        model_kwargs={"n_gpu_layers": -1},
        generation_kwargs={"max_tokens": 250, "temperature": 0.9},
        )

        reader = ExtractiveReader(model="deepset/roberta-base-squad2-distilled",)

        extractive_qa_pipeline = Pipeline()
        extractive_qa_pipeline.add_component("retriever", ChromaQueryTextRetriever(self.chroma_store))
        # extractive_qa_pipeline.add_component("reader",reader)
        extractive_qa_pipeline.add_component(instance=prompt_builder,   name="prompt_builder")
        extractive_qa_pipeline.add_component("llm", llm)
        extractive_qa_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")

        # extractive_qa_pipeline.connect("retriever.documents", "reader")
        extractive_qa_pipeline.connect("retriever", "prompt_builder.documents")     
        extractive_qa_pipeline.connect("prompt_builder", "llm")
        extractive_qa_pipeline.connect("llm.replies", "answer_builder.replies")
        extractive_qa_pipeline.connect("retriever", "answer_builder.documents")

        query = "who is Aditya  did Aditya Pursued his Masters from?"

        # Define the input data for the pipeline components
        input_data = {
            "retriever": {"query": query, "top_k": 3},
            # "reader": {"query": query},
            "prompt_builder": {"question": query},
            "answer_builder": {"query": query},
            # Use 'max_tokens' instead of 'max_new_tokens'
        }

        # Run the pipeline with the updated input data
        results = extractive_qa_pipeline.run(input_data)
        return results

        
        
        