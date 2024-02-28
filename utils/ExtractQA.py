from haystack.pipelines import DocumentSearchPipeline, ExtractiveQAPipeline
from haystack.nodes import JoinDocuments
from haystack import Pipeline




def ExtracQA(reader,retriever,query):
    qa_pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever)
    result = qa_pipeline.run(query=query,  params={"retriever": {"top_k": 3}, "reader": {"top_k": 5}})
    
    
    return result
    
    
    
    
def MultipleRetriever(reader,es_retriever,dpr_retriever,query):
    p = Pipeline()
    p.add_node(component=es_retriever, name="ESRetriever", inputs=["Query"])
    p.add_node(component=dpr_retriever, name="DPRRetriever", inputs=["Query"])
    p.add_node(component=JoinDocuments(join_mode="concatenate"), name="JoinResults", inputs=["ESRetriever", "DPRRetriever"])
    p.add_node(component=reader, name="QAReader", inputs=["JoinResults"])
    result = p.run(query=query, params={"ESRetriever": {"top_k": 10}, "DPRRetriever": {"top_k": 10}, "QAReader": {"top_k": 5}})    
    
    return result

