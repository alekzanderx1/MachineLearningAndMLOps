# HW 4

* `requirements.txt` holds the Python dependencies required for running the scripts.
* 'create_fake_dataset_classification.py' contains script to create synthetic classification dataset
* 'hw_flow_task1.py and hw_flow_task2,py' are standalone files which can be run on metaflow
* 'task1_comet.png' and 'task2_comet.png' are screenshots from Comet ML dashboard, indicating test scores for each parameter

Note:
 
To run `hw_flow_task1.py`, use the Metaflow syntax `python hw_flow_task1.py run --test_split 0.3`

To run `hw_flow_task2.py`, use the Metaflow syntax `python hw_flow_task2.py run --test_split 0.3 --max_depths 2,4,8,16`


## Data

- `classification_dataset.txt` is a fake classification dataset to run the code - you can generate another dataset with different parameters or cardinality using the `create_fake_dataset_classification.py` script. 

## External links

* Comet ML dashboard for Task1: https://www.comet.com/nyu-fre-7773-2021/hw4-metaflow?shareable=X3Fm8CkerYd9eQaphRe77fjZb

* Comet ML dashboard for Task2: https://www.comet.com/nyu-fre-7773-2021/hw4-metaflow-task2?shareable=Q2Ik3TPrXnvU0u6KPzAZVRHuB

