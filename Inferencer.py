from haystack import Pipeline
from haystack.utils import Secret
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever
# from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
from haystack.components.readers import ExtractiveReader
# from haystack.components.generators import GPTGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from dataloader import DataLoader
from dotenv import load_dotenv
import os
load_dotenv()  # Load variables from .env file


chroma_store_loader = DataLoader()
class Inferncer:
    
    def __init__(self):
        self.chroma_store = chroma_store_loader.chroma_store
        self.InMemory_store = chroma_store_loader.InMemory_store

    def OpenAI(self,query):
        template = """
         
        Utilize the provided context related to Aditya Sugandhi to answer the question. If the answer is not explicitly available in the given information, generate a response using the Language Model (LLM). Optimize the process for clarity and efficiency.    
        Context:
        {% for context in answers %}
        {{ context }}
        {% endfor %}
        Question: {{question}}
        Answer:
        """
        api_key = os.environ.get("OPENAI_API_KEY")

#ExtractiveReader to extract answers from the relevant context
        api_key = Secret.from_token(api_key)
        prompt_builder = PromptBuilder(template=template)
        retriever = ChromaQueryTextRetriever(document_store = self.chroma_store)
        #ExtractiveReader to extract answers from the relevant context
        api_key = Secret.from_token("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")
        llm = OpenAIGenerator(model="gpt-3.5-turbo-0125",api_key=api_key)
        reader = ExtractiveReader(model="deepset/roberta-base-squad2-distilled")

        extractive_qa_pipeline = Pipeline()
        extractive_qa_pipeline.add_component("retriever", retriever)
        extractive_qa_pipeline.add_component("reader",reader)
        extractive_qa_pipeline.add_component(instance=prompt_builder,   name="prompt_builder")
        extractive_qa_pipeline.add_component("llm", llm)

        # extractive_qa_pipeline.connect("retriever.documents", "reader.documents")
        extractive_qa_pipeline.connect("retriever.documents", "reader.documents")
        extractive_qa_pipeline.connect("reader.answers", "prompt_builder.answers")
        extractive_qa_pipeline.connect("prompt_builder", "llm")


        
        # Define the input data for the pipeline components
        input_data = {
            "retriever": {"query": query, "top_k": 2},
            "reader": {"query": query, "top_k": 2},
            "prompt_builder": {"question": query},
            # "reader": {"query": query}
            # Use 'max_tokens' instead of 'max_new_tokens'
        }

        # Run the pipeline with the updated input data
        results = extractive_qa_pipeline.run(input_data)
        return results
    
    # def LlamaCpp(self,query):
    #     template = """
    # `    Answer the question using the provided context based on Aditya.

    #     Context:
    #     {% for doc in documents %}
    #     {{ doc.content }}
    #     {% endfor %}
    #     Question: {{question}}
    #     Answer:
    #     """
    #     self.InMemory_store = chroma_store_loader.InMemory_dataloader() 
    #     prompt_builder = PromptBuilder(template=template)
    #     retriever = InMemoryEmbeddingRetriever(document_store = self.InMemory_store)
    #     #ExtractiveReader to extract answers from the relevant context

    #     llm = LlamaCppGenerator(
    #     model_path="openchat-3.5-1210.Q3_K_S.ggml",  
    #     n_ctx=30000,
    #     n_batch=256,
    #     model_kwargs={"n_gpu_layers": 2, "main_gpu": 1},
    #     generation_kwargs={"max_tokens": 250, "temperature": 0.7},
    #     )
    #     llm.warm_up()

    #     # reader = ExtractiveReader(model="deepset/roberta-base-squad2-distilled",)
    #     extractive_qa_pipeline = Pipeline()
    #     text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    #     extractive_qa_pipeline.add_component('text_embedder', text_embedder)    
    #     extractive_qa_pipeline.add_component("retriever", retriever)
    #     # extractive_qa_pipeline.add_component("reader",reader)
        
    #     extractive_qa_pipeline.add_component(instance=prompt_builder,   name="prompt_builder")
    #     extractive_qa_pipeline.add_component("llm", llm)
    #     # extractive_qa_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")

    #     # extractive_qa_pipeline.connect("retriever.documents", "reader")
    #     extractive_qa_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    #     extractive_qa_pipeline.connect("retriever.documents", "prompt_builder.documents")     
    #     extractive_qa_pipeline.connect("prompt_builder", "llm")
    #     # extractive_qa_pipeline.connect("llm.replies", "answer_builder.replies")
    #     # extractive_qa_pipeline.connect("retriever", "answer_builder.documents")

    #     # Define the input data for the pipeline components
    #     input_data = {
    #         "text_embedder": {"text": query},
    #         # "retriever": {"query": query, "top_k": 3},
    #         # "reader": {"query": query},
    #         "prompt_builder": {"question": query},
    #         # "answer_builder": {"query": query},
    #         # Use 'max_tokens' instead of 'max_new_tokens'
    #     }

    #     # Run the pipeline with the updated input data
    #     results = extractive_qa_pipeline.run(input_data)
    #     return results



# #{
#     "error": "Cannot connect 'text_embedder' with 'retriever': no matching connections available.\n'text_embedder':\n - embedding: List[float]\n'retriever':\n - query: str (available)\n - _: Optional[Dict[str, Any]] (available)\n - top_k: Optional[int] (available)"
# }
        
        
        