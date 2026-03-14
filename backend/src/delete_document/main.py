import os
import json
import boto3
from aws_lambda_powertools import Logger
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

DOCUMENT_TABLE = os.environ["DOCUMENT_TABLE"]
MEMORY_TABLE = os.environ["MEMORY_TABLE"]
BUCKET = os.environ["BUCKET"]
OPENSEARCH_ENDPOINT = os.environ["OPENSEARCH_ENDPOINT"]
OPENSEARCH_INDEX = os.environ["OPENSEARCH_INDEX"]
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

ddb = boto3.resource("dynamodb")
document_table = ddb.Table(DOCUMENT_TABLE)
memory_table = ddb.Table(MEMORY_TABLE)
s3 = boto3.client("s3")
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
    document_id = event["pathParameters"]["documentid"]

    response = document_table.get_item(
        Key={"userid": user_id, "documentid": document_id}
    )
    document = response["Item"]
    logger.info({"document": document})

    logger.info("Deleting memory table items")
    with memory_table.batch_writer() as batch:
        for item in document["conversations"]:
            batch.delete_item(Key={"userid": user_id, "SessionId": item["conversationid"]})

    document_table.delete_item(
        Key={"userid": user_id, "documentid": document_id}
    )

    logger.info("Deleting vectors from OpenSearch")
    try:
        client = get_opensearch_client()
        if client.indices.exists(index=OPENSEARCH_INDEX):
            client.delete_by_query(
                index=OPENSEARCH_INDEX,
                body={
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"user_id": user_id}},
                                {"term": {"document_id": document_id}},
                            ]
                        }
                    }
                },
            )
    except Exception as e:
        logger.warning("OpenSearch delete_by_query failed (index may not exist)", extra={"error": str(e)})

    logger.info("Deleting S3 object (PDF)")
    filename = document["filename"]
    pdf_key = f"{user_id}/{filename}/{filename}"
    s3.delete_objects(
        Bucket=BUCKET,
        Delete={"Objects": [{"Key": pdf_key}], "Quiet": True},
    )

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
        },
        "body": json.dumps({}, default=str),
    }
