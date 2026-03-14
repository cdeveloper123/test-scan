import os
import json
import boto3
from aws_lambda_powertools import Logger
from langchain_aws.embeddings import BedrockEmbeddings
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

DOCUMENT_TABLE = os.environ["DOCUMENT_TABLE"]
OPENSEARCH_ENDPOINT = os.environ["OPENSEARCH_ENDPOINT"]
OPENSEARCH_INDEX = os.environ["OPENSEARCH_INDEX"]
EMBEDDING_MODEL_ID = os.environ["EMBEDDING_MODEL_ID"]
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

SEARCH_SIZE = 20

ddb = boto3.resource("dynamodb")
document_table = ddb.Table(DOCUMENT_TABLE)
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


@logger.inject_lambda_context(log_event=True)
def lambda_handler(event, context):
    user_id = event["requestContext"]["authorizer"]["claims"]["sub"]
    query_params = event.get("queryStringParameters") or {}
    q = (query_params.get("q") or "").strip()
    if not q:
        return {
            "statusCode": 400,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
            },
            "body": json.dumps({"error": "Missing query parameter 'q'"}),
        }

    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION,
    )
    embeddings = BedrockEmbeddings(
        model_id=EMBEDDING_MODEL_ID,
        client=bedrock_runtime,
        region_name=AWS_REGION,
    )
    query_vector = embeddings.embed_query(q)

    client = get_opensearch_client()
    if not client.indices.exists(index=OPENSEARCH_INDEX):
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
            },
            "body": json.dumps({"results": []}),
        }

    body = {
        "size": SEARCH_SIZE,
        "query": {
            "bool": {
                "must": [{"knn": {"embedding": {"vector": query_vector, "k": SEARCH_SIZE}}}],
                "filter": {"bool": {"must": [{"term": {"user_id": user_id}}]}},
            }
        },
        "_source": ["document_id", "text", "page", "user_id"],
    }
    response = client.search(index=OPENSEARCH_INDEX, body=body)
    hits = response.get("hits", {}).get("hits", [])

    doc_ids = list({hit["_source"]["document_id"] for hit in hits})
    doc_map = {}
    if doc_ids:
        for doc_id in doc_ids:
            try:
                item = document_table.get_item(
                    Key={"userid": user_id, "documentid": doc_id}
                ).get("Item")
                if item:
                    doc_map[doc_id] = item.get("filename", doc_id)
            except Exception:
                doc_map[doc_id] = doc_id

    results = []
    for hit in hits:
        source = hit.get("_source", {})
        doc_id = source.get("document_id", "")
        results.append({
            "document_id": doc_id,
            "filename": doc_map.get(doc_id, doc_id),
            "text": source.get("text", "")[:500],
            "page": source.get("page", 0),
            "score": hit.get("_score"),
        })

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
        },
        "body": json.dumps({"results": results}, default=str),
    }
