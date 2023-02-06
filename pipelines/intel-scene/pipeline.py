"""Example workflow pipeline script for intel pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
)
from sagemaker.sklearn import SKLearn
from sagemaker.processing import FrameworkProcessor
from sagemaker.pytorch.processing import PyTorchProcessor

# from sagemaker.pytorch import PyTorchModel
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.steps import CacheConfig
from sagemaker.workflow.pipeline_context import PipelineSession

from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="AbalonePackageGroup",
    pipeline_name="AbalonePipeline",
    base_job_prefix="Abalone",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)
    
    # [START] Intel pipeline
    
    dvc_repo_url = ParameterString(
        name="DVCRepoURL",
        default_value="codecommit::ap-south-1://sagemaker-scene"
    )
    dvc_branch = ParameterString(
        name="DVCBranch",
        default_value="pipeline-processed-dataset"
    )
    
    input_dataset = ParameterString(
        name="InputDatasetZip",
        default_value="s3://sagemaker-ap-south-1-441249477288/dataset/intel.zip"
    )
    
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    model_name = ParameterString(name="Model", default_value="spnasnet_100")
    batch_size = ParameterString(name="Batch_Size", default_value="128")
    optimizer = ParameterString(name="Optimizer", default_value="torch.optim.Adam")
    learning_rate = ParameterString(name="Learning_Rate", default_value="0.0005")
    use_augmentation_pipeline = ParameterString(name="Augmentation_Pipeline", default_value="1")
    # train_data_s3 = ParameterString(name="Train_Dataset_S3", default_value=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri)
    # test_data_s3 = ParameterString(name="Test_Dataset_S3", default_value=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri)

    
    base_job_name = base_job_prefix
    
    # PREPROCESS STEP
    
    sklearn_processor = FrameworkProcessor(
        estimator_cls=SKLearn,
        framework_version="0.23-1",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        image_uri="441249477288.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-training:latest",
        base_job_name=f"{base_job_name}/preprocess-scene-dataset",
        sagemaker_session=pipeline_session,
        role=role,
        env={
            "DVC_REPO_URL": dvc_repo_url,
            "DVC_BRANCH": dvc_branch,
            "GIT_USER": "gokul-pv",
            "GIT_EMAIL": "25975535+gokul-pv@users.noreply.github.com"
        }
    )
    
    processing_step_args = sklearn_processor.run(
        code='pipelines/preprocess.py',
        source_dir=os.path.join(BASE_DIR, "sagemaker-kidu"),
        inputs=[
            ProcessingInput(
                input_name='data',
                source=input_dataset,
                destination='/opt/ml/processing/input'
            ),
            ProcessingInput(
                input_name='annotated_meta',
                source="s3://sagemaker-ap-south-1-441249477288/annotations/",
                destination='/opt/ml/processing/input/annotations/meta'
            ),
            ProcessingInput(
                input_name='annotated_data',
                source="s3://sagemaker-ap-south-1-441249477288/Predictions/",
                destination='/opt/ml/processing/input/annotations/data'
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/dataset/train"
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/dataset/test"
            ),
        ],
    )
    cache_config = CacheConfig(enable_caching=True, expire_after="30d")
    step_process = ProcessingStep(
        name="PreprocessSceneClassifierDataset",
        step_args=processing_step_args,
        cache_config=cache_config,
    )
    
    # TRAIN STEP
    
    tensorboard_output_config = TensorBoardOutputConfig(
        s3_output_path=f's3://{default_bucket}/sagemaker-scene-logs-pipeline',
        container_local_output_path='/opt/ml/output/tensorboard'
    )
    
    pt_estimator = PyTorch(
        base_job_name=f"{base_job_name}/training-scene-pipeline",
        image_uri = "441249477288.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-training:latest",
        source_dir=os.path.join(BASE_DIR, "sagemaker-kidu"),
        entry_point="pipelines/train_pipeline.py",
        sagemaker_session=pipeline_session,
        role=role,
        instance_count=1,
        instance_type="ml.g4dn.xlarge",
        tensorboard_output_config=tensorboard_output_config,
        use_spot_instances=True,
        max_wait=7200,
        max_run=5000,
        environment={
            "GIT_USER": "gokul-pv",
            "GIT_EMAIL": "25975535+gokul-pv@users.noreply.github.com",
            "MODEL": model_name,
            "BATCH_SIZE": batch_size,
            "OPTIMIZER": optimizer,
            "LR": learning_rate,
            "AUGMENTATION": use_augmentation_pipeline
        }
    )
    
    estimator_step_args = pt_estimator.fit({
        'train': TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                "train"
            ].S3Output.S3Uri,
        ),
        'test': TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                "test"
            ].S3Output.S3Uri,
        )
    })
    
    step_train = TrainingStep(
        name="TrainSceneClassifier",
        step_args=estimator_step_args,
    )
    
    # EVAL STEP
    
    pytorch_processor = PyTorchProcessor(
        image_uri = "441249477288.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-training:latest",
        framework_version='1.11.0',
        py_version="py38",
        role=role,
        sagemaker_session=pipeline_session,
        instance_type="ml.m5.4xlarge",
        instance_count=1,
        base_job_name=f'{base_job_name}/eval-scene-classifier',
        env={
        "BATCH_SIZE": batch_size,
        "MODEL": model_name,

        }
    )

    
    eval_step_args = pytorch_processor.run(
        code='pipelines/evaluate.py',
        source_dir=os.path.join(BASE_DIR, "sagemaker-kidu"),
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                destination="/opt/ml/processing/train",
            ),
            ProcessingInput(
                source="s3://sagemaker-ap-south-1-441249477288/old-data/prediction/",
                destination="/opt/ml/processing/prediction",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
    )
    
    evaluation_report = PropertyFile(
        name="SceneClassifierEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="EvaluateSceneClassifierModel",
        step_args=eval_step_args,
        property_files=[evaluation_report],
    )

    # MODEL REGISTER STEP
    
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )
    
    # model = PyTorchModel(
    #     entry_point="pipelines/infer.py",
    #     source_dir=os.path.join(BASE_DIR, "sagemaker-kidu"),
    #     image_uri = "441249477288.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-inference",
    #     sagemaker_session=pipeline_session,
    #     role=role,
    #     model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    #     framework_version="1.11.0",
    # )
    model = Model(
        image_uri = "441249477288.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-inference:latest",
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        sagemaker_session=pipeline_session,
        predictor_cls=Predictor
    )

    model_step_args = model.register(
        content_types=["file-path/raw-bytes"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium", "ml.t2.large"],
        transform_instances=["ml.m4.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    
    step_register = ModelStep(
        name="RegisterSceneClassifierModel",
        step_args=model_step_args,
    )
    
    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="multiclass_classification_metrics.accuracy.value"
        ),
        right=0.6,
    )

    step_cond = ConditionStep(
        name="CheckAccuracySceneClassifierEvaluation",
        conditions=[cond_gte],
        if_steps=[step_register],
        else_steps=[],
    )

    
    # [END] intel pipeline

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            dvc_repo_url,
            dvc_branch,
            input_dataset,
            model_approval_status,
            model_name,
            batch_size,
            optimizer,
            learning_rate,
            use_augmentation_pipeline,
        ],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=pipeline_session,
    )
    
    return pipeline
