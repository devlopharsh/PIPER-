from __future__ import annotations
import argparse
import sagemaker
from pathlib import Path
from sagemaker.pytorch import PyTorchModel
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker.s3 import S3Uploader
from boto3.session import Session


def main():
    ap = argparse.ArgumentParser(description="Upload model.tar.gz and deploy a GRU classifier endpoint on SageMaker.")
    ap.add_argument("--model_tar", type=str, required=True, help="path to model.tar.gz (must include server/inference.py)")
    ap.add_argument("--s3_bucket", type=str, required=True, help="S3 bucket to upload model artifacts")
    ap.add_argument("--s3_prefix", type=str, default="leak-gru", help="S3 prefix for model artifacts")
    ap.add_argument("--role_arn", type=str, required=True, help="SageMaker execution role ARN")
    ap.add_argument("--endpoint_name", type=str, default="leak-gru-endpoint")
    ap.add_argument("--region", type=str, default="eu-central-1")
    ap.add_argument("--instance_type", type=str, default="ml.m5.large", help="used when NOT using --serverless")
    ap.add_argument("--serverless", action="store_true", help="deploy as a serverless endpoint")
    ap.add_argument("--serverless_mem", type=int, default=2048, help="serverless memory in MB")
    ap.add_argument("--serverless_concurrency", type=int, default=2, help="max concurrent invocations for serverless")
    ap.add_argument("--thr_override", type=float, default=None, help="optional threshold override via endpoint env var")
    ap.add_argument("--framework_version", type=str, default="2.1", help="PyTorch framework version")
    ap.add_argument("--py_version", type=str, default="py310", help="Python version tag for the container")
    args = ap.parse_args()

    model_tar_path = Path(args.model_tar)
    if not model_tar_path.exists():
        raise FileNotFoundError(f"model_tar not found: {model_tar_path}")

    # Create sessions (boto and SageMaker) pinned to region
    boto_sess = Session(region_name=args.region)
    sm_session = sagemaker.Session(boto_session=boto_sess)

    # Upload model.tar.gz to S3
    # NOTE: SageMaker expects your tar to contain your artifacts AND the source_dir files.
    #       Here, 'server/inference.py' is referenced below as the model's entry point.
    s3_uri = S3Uploader.upload(
        str(model_tar_path),
        f"s3://{args.s3_bucket}/{args.s3_prefix}",
        sagemaker_session=sm_session,
    )
    print("[UPLOAD] Model uploaded to:", s3_uri)

    # Optional: pass a global threshold override at the endpoint level
    env = {}
    if args.thr_override is not None:
        env["THR_OVERRIDE"] = str(args.thr_override)

    # Define the model object backed by the built-in PyTorch serving container
    model = PyTorchModel(
        role=args.role_arn,
        model_data=s3_uri,
        entry_point="inference.py",     # this file must exist under 'source_dir'
        source_dir="server",            # folder inside model.tar.gz
        framework_version=args.framework_version,
        py_version=args.py_version,
        env=env,
        sagemaker_session=sm_session,
    )

    # Deploy either serverless or on a provisioned instance
    if args.serverless:
        # Serverless is billed per invocation and scales to zero—great for low/variable traffic.
        serverless_config = ServerlessInferenceConfig(
            memory_size_in_mb=args.serverless_mem,
            max_concurrency=args.serverless_concurrency,
        )
        predictor = model.deploy(
            endpoint_name=args.endpoint_name,
            serverless_inference_config=serverless_config,
        )
    else:
        # Provisioned endpoint—lower latency and consistent capacity for steady workloads.
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=args.instance_type,
            endpoint_name=args.endpoint_name,
        )

    print("Endpoint created:", predictor.endpoint_name)

if __name__ == "__main__":
    main()
