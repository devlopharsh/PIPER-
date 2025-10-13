from __future__ import annotations
import argparse, boto3, sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.serverless import ServerlessInferenceConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tar", type=str, required=True, help="path to model.tar.gz")
    ap.add_argument("--s3_bucket", type=str, required=True)
    ap.add_argument("--s3_prefix", type=str, default="leak-gru")
    ap.add_argument("--role_arn", type=str, required=True, help="SageMaker execution role ARN")
    ap.add_argument("--endpoint_name", type=str, default="leak-gru-endpoint")
    ap.add_argument("--region", type=str, default="eu-central-1")
    ap.add_argument("--instance_type", type=str, default="ml.m5.large")
    ap.add_argument("--serverless", action="store_true", help="use serverless endpoint")
    # προσθέτουμε CLI flags
    ap.add_argument("--serverless_mem", type=int, default=2048)
    ap.add_argument("--serverless_concurrency", type=int, default=2)


    ap.add_argument("--thr_override", type=float, default=None, help="override threshold at endpoint env")
    ap.add_argument("--framework_version", type=str, default="2.1", help="PyTorch framework version")
    ap.add_argument("--py_version", type=str, default="py310", help="Python version tag")
    args = ap.parse_args()

    # Sessions
    boto_sess = boto3.session.Session(region_name=args.region)
    sm_session = sagemaker.Session(boto_session=boto_sess)

    # Upload model.tar.gz to S3
    s3_uri = sagemaker.s3.S3Uploader.upload(
    args.model_tar,
    f"s3://{args.s3_bucket}/{args.s3_prefix}",
    sagemaker_session=sm_session,   # <= ΑΥΤΟ
    )

    print("[UPLOAD] model to:", s3_uri)

    # περιβάλλον για threshold override (προαιρετικό)
    env = {}
    if args.thr_override is not None:
        env["THR_OVERRIDE"] = str(args.thr_override)

    # φτιάξε PyTorchModel (χρησιμοποιεί built-in PyTorch container)
    model = PyTorchModel(
        role=args.role_arn,
        model_data=s3_uri,
        entry_point="inference.py",
        source_dir="server",            # ο ΦΑΚΕΛΟΣ που έχει το inference.py
        framework_version=args.framework_version,
        py_version=args.py_version,
        env=env,
        sagemaker_session=sm_session,
    )

    if args.serverless:
        serverless_config = ServerlessInferenceConfig(
            memory_size_in_mb=args.serverless_mem,
            max_concurrency=args.serverless_concurrency
        )
        predictor = model.deploy(
            endpoint_name=args.endpoint_name,
            serverless_inference_config=serverless_config
        )
    else:
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=args.instance_type,
            endpoint_name=args.endpoint_name
        )

    print("Endpoint created:", predictor.endpoint_name)

if __name__ == "__main__":
    main()
