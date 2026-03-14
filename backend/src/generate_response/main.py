import os
import json
import boto3
from aws_lambda_powertools import Logger
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain_aws.chat_models import ChatBedrock
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from typing import List

MEMORY_TABLE = os.environ["MEMORY_TABLE"]
MODEL_ID = os.environ["MODEL_ID"]
EMBEDDING_MODEL_ID = os.environ["EMBEDDING_MODEL_ID"]
OPENSEARCH_ENDPOINT = os.environ["OPENSEARCH_ENDPOINT"]
OPENSEARCH_INDEX = os.environ["OPENSEARCH_INDEX"]
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

K = 5

logger = Logger()


def get_opensearch_client():
    host = OPENSEARCH_ENDPOINT.replace("https://", "").replace("http://", "").rstrip("/")
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        AWS_REGION,
        "aoss",
        session_token=credentials.token,
    )
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )


class OpenSearchRetriever(BaseRetriever):
    client: OpenSearch = Field(exclude=True)
    index_name: str
    embeddings: BedrockEmbeddings
    user_id: str
    document_id: str
    k: int = K

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        query_vector = self.embeddings.embed_query(query)
        body = {
            "size": self.k,
            "query": {
                "bool": {
                    "must": [
                        {"knn": {"embedding": {"vector": query_vector, "k": self.k}}}
                    ],
                    "filter": {
                        "bool": {
                            "must": [
                                {"term": {"user_id": self.user_id}},
                                {"term": {"document_id": self.document_id}},
                            ]
                        }
                    },
                }
            },
        }
        response = self.client.search(index=self.index_name, body=body)
        docs = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            docs.append(
                Document(
                    page_content=source.get("text", ""),
                    metadata={"page": source.get("page", 0), "chunk_index": source.get("chunk_index", 0)},
                )
            )
        return docs


def get_embeddings():
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION,
    )
    return BedrockEmbeddings(
        model_id=EMBEDDING_MODEL_ID,
        client=bedrock_runtime,
        region_name=AWS_REGION,
    )


def create_memory(user_id, conversation_id):
    message_history = DynamoDBChatMessageHistory(
        table_name=MEMORY_TABLE,
        session_id=conversation_id,
        key={"userid": user_id, "SessionId": conversation_id},
    )
    return ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=message_history,
        input_key="question",
        output_key="answer",
        return_messages=True,
    )


def bedrock_chain(retriever, memory, human_input):
    chat = ChatBedrock(model_id=MODEL_ID, model_kwargs={"temperature": 0.0})
    chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )
    return chain.invoke({"question": human_input})


@logger.inject_lambda_context(log_event=True)
def lambda_handler(event, context):
    event_body = json.loads(event["body"])
    human_input = event_body["prompt"]
    conversation_id = event["pathParameters"]["conversationid"]
    document_id = event["pathParameters"]["documentid"]
    user = event["requestContext"]["authorizer"]["claims"]["sub"]

    embeddings = get_embeddings()
    client = get_opensearch_client()
    retriever = OpenSearchRetriever(
        client=client,
        index_name=OPENSEARCH_INDEX,
        embeddings=embeddings,
        user_id=user,
        document_id=document_id,
        k=K,
    )
    memory = create_memory(user, conversation_id)

    response = bedrock_chain(retriever, memory, human_input)
    if not response:
        raise ValueError(f"Unsupported model ID: {MODEL_ID}")

    logger.info(str(response["answer"]))

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
        },
        "body": json.dumps(response["answer"]),
    }
