from pathlib import Path
import os
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder,SentenceTransformersTextEmbedder
from haystack.components.converters import PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
class DataLoader:
    
    def __init__(self):
        self.chroma_store = ChromaDocumentStore()
        self.InMemory_store = InMemoryDocumentStore()
    
    def dataloader(self):
        HERE = Path(os.getcwd())


        data_path = HERE / "data"
        file_paths = [str(data_path / name) for name in os.listdir(data_path)]

        

        pipeline = Pipeline()
        pipeline.add_component("FileTypeRouter", FileTypeRouter(mime_types=["text/plain", "application/pdf"]))
        pipeline.add_component("TextFileConverter", TextFileToDocument())
        pipeline.add_component("PdfFileConverter", PyPDFToDocument())

        pipeline.add_component("Joiner", DocumentJoiner())
        pipeline.add_component("Cleaner", DocumentCleaner())
        pipeline.add_component("Splitter", DocumentSplitter(split_by="sentence", split_length=250, split_overlap=30))
        # pipeline.add_component("TextEmbedder", SentenceTransformersTextEmbedder())
        pipeline.add_component("Embedder", SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))

        pipeline.add_component("Writer", DocumentWriter(document_store=self.chroma_store))

        pipeline.connect("FileTypeRouter.text/plain", "TextFileConverter.sources")
        pipeline.connect("FileTypeRouter.application/pdf", "PdfFileConverter.sources")
        pipeline.connect("TextFileConverter.documents", "Joiner.documents")
        pipeline.connect("PdfFileConverter.documents", "Joiner.documents")
        pipeline.connect("Joiner.documents", "Cleaner.documents")
        pipeline.connect("Cleaner.documents", "Splitter.documents")
        pipeline.connect("Splitter.documents", "Embedder.documents")
        # pipeline.connect("TextEmbedder.embeddings", "Embedder.documents")
        pipeline.connect("Embedder.documents", "Writer.documents")



        pipeline.run(
            {"FileTypeRouter": {"sources": file_paths}},
         
        )
        return self.chroma_store
    
    
    def InMemory_dataloader(self):
        HERE = Path(os.getcwd())


        data_path = HERE / "data"
        file_paths = [str(data_path / name) for name in os.listdir(data_path)]

        

        pipeline = Pipeline()
        pipeline.add_component("FileTypeRouter", FileTypeRouter(mime_types=["text/plain", "application/pdf"]))
        pipeline.add_component("TextFileConverter", TextFileToDocument())
        pipeline.add_component("PdfFileConverter", PyPDFToDocument())

        pipeline.add_component("Joiner", DocumentJoiner())
        pipeline.add_component("Cleaner", DocumentCleaner())
        pipeline.add_component("Splitter", DocumentSplitter(split_by="sentence", split_length=250, split_overlap=30))
        # pipeline.add_component("TextEmbedder", SentenceTransformersTextEmbedder())
        pipeline.add_component("Embedder", SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))

        pipeline.add_component("Writer", DocumentWriter(document_store=self.InMemory_store))

        pipeline.connect("FileTypeRouter.text/plain", "TextFileConverter.sources")
        pipeline.connect("FileTypeRouter.application/pdf", "PdfFileConverter.sources")
        pipeline.connect("TextFileConverter.documents", "Joiner.documents")
        pipeline.connect("PdfFileConverter.documents", "Joiner.documents")
        pipeline.connect("Joiner.documents", "Cleaner.documents")
        pipeline.connect("Cleaner.documents", "Splitter.documents")
        pipeline.connect("Splitter.documents", "Embedder.documents")
        # pipeline.connect("TextEmbedder.embeddings", "Embedder.documents")
        pipeline.connect("Embedder.documents", "Writer.documents")



        pipeline.run(
            {"FileTypeRouter": {"sources": file_paths}},
         
        )
        return self.InMemory_store
    
    
    def get_chroma_store(self):
        return self.chroma_store
    
    def get_InMemory_store(self):
        return self.InMemory_store
    
