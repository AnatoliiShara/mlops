from kfp import compiler
from pipelines.kubeflow.inference_pipeline import inference_pipeline

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=inference_pipeline,
        package_path="inference_pipeline.yaml"
    )
