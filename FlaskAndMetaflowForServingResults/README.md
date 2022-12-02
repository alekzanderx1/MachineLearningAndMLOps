# HW 4

* `requirements.txt` holds the Python dependencies required for running the scripts.
* 'create_fake_loans_dataset.py' contains script to create synthetic Loan Approval classification dataset
* 'loans_dataset.csv' is output of above script
* 'hw_flow.py' is standalone metaflow file containing DAG to Train and Test the Model
* 'app.py' and '/templates' folder contain Flask and Streamlit code to launch a web app to serve results of the latest model run

Steps:
 
1. Run `hw_flow.py`, use the Metaflow syntax `python hw_flow.py run --test_split 0.2 --max_depths 2,4,8,16 --with card`
	Note: This may give errors the first time due to metaflow issues, please run again in case errors are seen for card generation
	
2. Update location of ./metaflow folder in 'app.py' line 17 based on where your system generates metaflow artifacts

3. To launch Flash app use `python app.py`, you app should be available at http://127.0.0.1:5000 



## Data

- `loans_dataset.csv` is a fake classification dataset to run the code - you can generate another dataset with different parameters or cardinality using the `create_fake_loans_dataset.py` script. 


