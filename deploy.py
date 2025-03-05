import os
import uuid
from google.cloud import aiplatform

def deploy_to_vertex_ai(model_path):
    aiplatform.init(project=os.getenv('GCP_PROJECT_ID'))

    deployed_model = aiplatform.Model.upload(
        display_name=f"review_analyzer_{uuid.uuid4()}",
        artifact_uri=model_path,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch_gpu.1-11:latest"
    )

    endpoint = deployed_model.deploy(machine_type="n1-standard-4")
    return endpoint
