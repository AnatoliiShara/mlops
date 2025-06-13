from kfp import compiler
from pipelines.kubeflow.train_pipeline import training_pipeline

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path="train_pipeline.yaml"
    )
