from haystack.nodes import EmbeddingRetriever, BM25Retriever
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import JoinDocuments, SentenceTransformersRanker
from haystack.pipelines import Pipeline
from haystack.nodes import PreProcessor,PromptNode
from haystack.nodes import PromptNode, PromptTemplate, AnswerParser, BM25Retriever
from haystack.pipelines import Pipeline
from haystack.agents.memory import ConversationSummaryMemory
from haystack.agents import AgentStep, Agent
from haystack.agents.base import Agent, ToolsManager
from haystack.agents import Tool


import pandas as pd
import numpy as np
from openai import OpenAI
import os
import streamlit as st

openai_api_key = os.environ["API_KEY"]

file_path = 'Rieker_SUMMERANDWINTER_DATA.xlsx'

Rieker_Database = pd.read_excel(file_path)

seed_value = 42
np.random.seed(seed_value)

# Your original code
df_groupByColor_Rieker = Rieker_Database.groupby('Main_Color', group_keys=False).apply(lambda x: x.sample(10, replace=True, random_state=seed_value))
df_groupByShoeType_Rieker = Rieker_Database.groupby('main_category', group_keys=False).apply(lambda x: x.sample(10, replace=True, random_state=seed_value))
df_groupByGender_Rieker = Rieker_Database.groupby('Warengruppe', group_keys=False).apply(lambda x: x.sample(10, replace=True, random_state=seed_value))
df_groupBySaison_Rieker = Rieker_Database.groupby('Saison_Catch', group_keys=False).apply(lambda x: x.sample(10, replace=True, random_state=seed_value))
df_groupByMaterial_Rieker = Rieker_Database.groupby('EAS Material', group_keys=False).apply(lambda x: x.sample(10, replace=True, random_state=seed_value))

result_df = pd.concat([df_groupByColor_Rieker, df_groupByShoeType_Rieker, df_groupByGender_Rieker, df_groupBySaison_Rieker, df_groupByMaterial_Rieker], ignore_index=True)
result_df = result_df.drop_duplicates(subset='ID', keep='first')
Rieker_Database = result_df

docs = []
for index, row in Rieker_Database.iterrows():
    document = {
        'content': ', '.join(str(value) for value in row),
        'meta': {'ID': row['ID'],'Main_Color': row['Main_Color'], 'Main_Category': row['main_category'], 'Gender': row['Warengruppe'], 'Saison': row['Saison_Catch'],'Main_Material': row['EAS Material']}
    }
    docs.append(document)



#Gerade unklar ob preprocessor wirklich benutzt wird
from haystack.nodes import PreProcessor
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="word",
    split_length=512,    #Vlt hier die Länge auf 100 setzen um die
    split_overlap=32,
    split_respect_sentence_boundary=True, # Stellt sicher dass das Dokument nicht mitten im Satz aufgeteilt wird, um semantische Bedeutung für semantische Suche aufrecht erhalten werden kann
)


docs_to_index = preprocessor.process(docs)

document_store = InMemoryDocumentStore(use_bm25=True, embedding_dim=384)



sparse_retriever = BM25Retriever(document_store=document_store)
dense_retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",   # Jeder Dense Retriever hat im Normalfall eine begrenzte Anzahl an Tokens die er gleichzeitig verarbeiten kann-> im Normfall 256
                                                                # Welche Dense Retriever Methode wird verwendet? It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.
    use_gpu=True,
    scale_score=False,
)

document_store.delete_documents()
document_store.write_documents(docs_to_index)   #Hier evtl docs_to_index einfügen falls preprocessor wirklich nötig
document_store.update_embeddings(retriever=dense_retriever) #Hier Laaaaange Laufzeit Batches die entschlüsselt werden -> dauert hier die Indexierung so lange?

join_documents = JoinDocuments(join_mode="concatenate")     # Dabei werden beim Hybrid Retrieval alle Dokumente einfach an die Liste drangehängt ohne die Reihenfolge zu verändern
rerank = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2")

prompt_template = PromptTemplate(
    prompt="""
    Documents:{join(documents)}
    User_Input:{query}
    Answer:
    """,
    output_parser=AnswerParser(),
)

prompt_node = PromptNode(
    model_name_or_path="gpt-4-1106-preview", api_key=openai_api_key, default_prompt_template=prompt_template
)

pipeline = Pipeline()
pipeline.add_node(component=sparse_retriever, name="SparseRetriever", inputs=["Query"])
pipeline.add_node(component=dense_retriever, name="DenseRetriever", inputs=["Query"])
pipeline.add_node(component=join_documents, name="JoinDocuments", inputs=["SparseRetriever", "DenseRetriever"])
pipeline.add_node(component=rerank, name="ReRanker", inputs=["JoinDocuments"])
pipeline.add_node(component=prompt_node, name="prompt_node", inputs=["ReRanker"])




search_tool = Tool(
    name="shoe_hybrid_retrieval_pipeline",
    pipeline_or_node=pipeline,
    description="useful for when you need to adivse the User to find the right Shoe out of a shoe Database",
    output_variable="answers"
)

agent_prompt_node = PromptNode(
    "gpt-3.5-turbo",
    api_key=openai_api_key,
    max_length=1024,
    stop_words=["Observation:"],
    model_kwargs={"temperature": 0.5},
)


memory_prompt_node = PromptNode(
    "philschmid/bart-large-cnn-samsum", max_length=1024, model_kwargs={"task_name": "text2text-generation"}
)
memory = ConversationSummaryMemory(memory_prompt_node, prompt_template="{chat_transcript}")

agent_prompt = """
The following is the previous conversation between a human and The AI Agent:
{memory}
Question: {query}

In the following conversation, a human user interacts with an AI Agent. The human gives Informations about shoes and the Agent should try the best fitting shoes to this Descripton.
The AI Agent is a polite Consultant how should help the User finde the right shoes out of a databse. The Tool the Agent is using provides a list of shoes that are matching to the User Input.
Ask the User about Informations that are useful to find the right shoes out of the Shoe Database. 
You should only ask the User about more Informations that are Attributes in list of Documents that provided by the Agents Tool {tool_names_with_descriptions}.

The AI Agent must use the available tools to find the up-to-date information. The final answer to the question should be truthfully based solely on the output of the tools. The AI Agent should ignore its knowledge when answering the questions.
The AI Agent has access to these tools:
{tool_names_with_descriptions}


Thought:
{transcript}
"""


def resolver_function(query, agent, agent_step):
    return {
        "query": query,
        "tool_names_with_descriptions": agent.tm.get_tool_names_with_descriptions(),
        "transcript": agent_step.transcript,
        "memory": agent.memory.load(),
    }



conversational_agent = Agent(
    agent_prompt_node,
    prompt_template=agent_prompt,
    prompt_parameters_resolver=resolver_function,
    memory=memory,
    tools_manager=ToolsManager([search_tool]),
)

conversational_agent.run("Ich hätte gerne rote Schuhe ")




