from flytekitplugins.domino.helpers import Input, Output, run_domino_job_task
from flytekitplugins.domino.task import DatasetSnapshot
from flytekit import workflow
from flytekit.types.file import FlyteFile
from flytekit.types.directory import FlyteDirectory
from typing import TypeVar, NamedTuple

final_outputs = NamedTuple('final_outputs', model=FlyteFile[TypeVar('pkl')])

@workflow
def model_training(data_path_a: str, data_path_b: str) -> final_outputs: 
    '''
    Sample data preparation and training flow. This flow:
    
        1. Loads two datasets in from different sources
        2. Merges the data together
        3. Does some data preprocessing
        4. Trains a model using the processed data

    To run this flow, execute the following line in the terminal

    pyflyte run --remote flow.py model_training --data_path_a /mnt/code/data/datasetA.csv --data_path_b /mnt/code/data/datasetB.csv

    :param data_path_a: Path to datasetA
    :param data_path_b: Path to datasetB 
    :return: The training results as a model
    '''

    task1 = run_domino_job_task(
        flyte_task_name='Load Data A',
        command='python /mnt/code/scripts/load-data-A.py',
        hardware_tier_name='Small',
        inputs=[
            Input(name='data_path', type=str, value=data_path_a)
        ],
        output_specs=[
            Output(name='datasetA', type=FlyteFile[TypeVar('csv')])
        ],
        use_project_defaults_for_omitted=True
    )

    task2 = run_domino_job_task(
        flyte_task_name='Load Data B',
        command='python /mnt/code/scripts/load-data-B.py',
        hardware_tier_name='Small',
        inputs=[
            Input(name='data_path', type=str, value=data_path_b)
        ],
        output_specs=[
            Output(name='datasetB', type=FlyteFile[TypeVar('csv')])
        ],
        use_project_defaults_for_omitted=True
    )

    task3 = run_domino_job_task(
        flyte_task_name='Merge Data',
        command='python /mnt/code/scripts/merge-data.py',
        hardware_tier_name='Medium',
        inputs=[
            Input(name='datasetA', type=FlyteFile[TypeVar('csv')], value=task1['datasetA']),
            Input(name='datasetB', type=FlyteFile[TypeVar('csv')], value=task2['datasetB'])
        ],
        output_specs=[
            Output(name='merged_data', type=FlyteFile[TypeVar('csv')])
        ],
        use_project_defaults_for_omitted=True
    )

    task4 = run_domino_job_task(
        flyte_task_name='Process Data',
        command='python /mnt/code/scripts/process-data.py',
        hardware_tier_name='Medium',
        inputs=[
            Input(name='merged_data', type=FlyteFile[TypeVar('csv')], value=task3['merged_data'])
        ],
        output_specs=[
            Output(name='processed_data', type=FlyteFile[TypeVar('csv')])
        ],
        use_project_defaults_for_omitted=True
    )

    task5 = run_domino_job_task(
        flyte_task_name='Train Model',
        command='python /mnt/code/scripts/train-model.py',
        hardware_tier_name='Large',
        inputs=[
            Input(name='processed_data', type=FlyteFile[TypeVar('csv')], value=task4['processed_data']),
            Input(name='num_estimators', type=int, value=100)
        ],
        output_specs=[
            Output(name='model', type=FlyteFile[TypeVar('pkl')])
        ],
        use_project_defaults_for_omitted=True
    )

    return final_outputs(model=task5['model'])
