import os
import json
import boto3
from aws_lambda_powertools import Logger
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

DOCUMENT_TABLE = os.environ["DOCUMENT_TABLE"]
BUCKET = os.environ["BUCKET"]
EMBEDDING_MODEL_ID = os.environ["EMBEDDING_MODEL_ID"]
OPENSEARCH_ENDPOINT = os.environ["OPENSEARCH_ENDPOINT"]
OPENSEARCH_INDEX = os.environ["OPENSEARCH_INDEX"]
DOCUMENT_READY_TOPIC_ARN = os.environ["DOCUMENT_READY_TOPIC_ARN"]
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Titan embed text v2 default dimension
EMBEDDING_DIMENSION = 1024

s3 = boto3.client("s3")
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


def ensure_index(client):
    if not client.indices.exists(index=OPENSEARCH_INDEX):
        body = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "user_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "text": {"type": "text"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": EMBEDDING_DIMENSION,
                        "space_type": "l2",
                    },
                    "page": {"type": "integer"},
                    "chunk_index": {"type": "integer"},
                }
            },
        }
        client.indices.create(index=OPENSEARCH_INDEX, body=body)
        logger.info("Created OpenSearch index", extra={"index": OPENSEARCH_INDEX})


def set_doc_status(user_id, document_id, status):
    document_table.update_item(
        Key={"userid": user_id, "documentid": document_id},
        UpdateExpression="SET docstatus = :docstatus",
        ExpressionAttributeValues={":docstatus": status},
    )


def publish_document_ready(user_id, document_id):
    sns = boto3.client("sns", region_name=AWS_REGION)
    sns.publish(
        TopicArn=DOCUMENT_READY_TOPIC_ARN,
        Message=json.dumps({"user_id": user_id, "document_id": document_id, "status": "READY"}),
        Subject="Document ready for chat",
    )
    logger.info("Published document ready", extra={"document_id": document_id})


@logger.inject_lambda_context(log_event=True)
def lambda_handler(event, context):
    event_body = json.loads(event["Records"][0]["body"])
    document_id = event_body["documentid"]
    user_id = event_body["user"]
    key = event_body["key"]
    file_name_full = key.split("/")[-1]

    set_doc_status(user_id, document_id, "PROCESSING")

    s3.download_file(BUCKET, key, f"/tmp/{file_name_full}")

    loader = PyPDFLoader(f"/tmp/{file_name_full}")
    raw_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    docs = text_splitter.split_documents(raw_docs)

    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION,
    )
    embeddings_model = BedrockEmbeddings(
        model_id=EMBEDDING_MODEL_ID,
        client=bedrock_runtime,
        region_name=AWS_REGION,
    )

    vectors = embeddings_model.embed_documents([d.page_content for d in docs])

    client = get_opensearch_client()
    ensure_index(client)

    bulk_body = []
    for i, (doc, vec) in enumerate(zip(docs, vectors)):
        page = doc.metadata.get("page", 0)
        if isinstance(page, str):
            try:
                page = int(page)
            except (ValueError, TypeError):
                page = 0
        bulk_body.append({"index": {"_index": OPENSEARCH_INDEX}})
        bulk_body.append({
            "user_id": user_id,
            "document_id": document_id,
            "text": doc.page_content,
            "embedding": vec,
            "page": page,
            "chunk_index": i,
        })
    if bulk_body:
        client.bulk(body=bulk_body, refresh=True)

    set_doc_status(user_id, document_id, "READY")
    publish_document_ready(user_id, document_id)
