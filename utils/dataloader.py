import os
from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.writers import DocumentWriter

from haystack_integrations.document_stores.chroma import ChromaDocumentStore





def load_data():
    file_paths = ["data" / Path(name) for name in os.listdir("data")]

    # Chroma is used in-memory so we use the same instances in the two pipelines below
    document_store = ChromaDocumentStore()

    indexing = Pipeline()
    indexing.add_component("converter", TextFileToDocument())
    indexing.add_component("writer", DocumentWriter(document_store))
    indexing.connect("converter", "writer")
    indexing.run({"converter": {"sources": file_paths}})

    return document_store



