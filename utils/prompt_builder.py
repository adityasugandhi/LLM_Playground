from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever
from haystack.components.generators import HuggingFaceTGIGenerator
from haystack.components.builders import PromptBuilder
from haystack.agents.memory import ConversationSummaryMemory
from dataloader import load_data
from hayst
prompt = """
Answer the query based on the provided context for Aditya.
If the context does not contain the answer, say 'Answer not found'.
Context:
{% for doc in documents %}
  {{ doc.content }}
{% endfor %}
query: {{query}}
Answer:
"""
prompt_builder = PromptBuilder(template=prompt)

llm = HuggingFaceTGIGenerator(model="mistralai/Mixtral-8x7B-Instruct-v0.1")
llm.warm_up()
retriever = ChromaQueryTextRetriever(load_data())

querying = Pipeline()
querying.add_component("retriever", retriever)
querying.add_component("prompt_builder", prompt_builder)
querying.add_component("llm", llm)

querying.connect("retriever.documents", "prompt_builder.documents")
querying.connect("prompt_builder", "llm")