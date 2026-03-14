import os
import json
import boto3
from aws_lambda_powertools import Logger

DOCUMENT_READY_TOPIC_ARN = os.environ["DOCUMENT_READY_TOPIC_ARN"]
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

sns = boto3.client("sns", region_name=AWS_REGION)
logger = Logger()


@logger.inject_lambda_context(log_event=True)
def lambda_handler(event, context):
    for record in event.get("Records", []):
        try:
            body = json.loads(record.get("body", "{}"))
            document_id = body.get("documentid", "unknown")
            user_id = body.get("user", "unknown")
            key = body.get("key", "")
            logger.warning(
                "Embedding job failed (DLQ)",
                extra={"document_id": document_id, "user_id": user_id, "key": key},
            )
            sns.publish(
                TopicArn=DOCUMENT_READY_TOPIC_ARN,
                Subject="[Alert] Document embedding failed",
                Message=json.dumps({
                    "type": "embedding_failed",
                    "document_id": document_id,
                    "user_id": user_id,
                    "s3_key": key,
                    "message": "Document processing failed after retries and was sent to DLQ.",
                }),
            )
        except Exception as e:
            logger.exception("Failed to process DLQ record", extra={"error": str(e)})
    return {"statusCode": 200}
