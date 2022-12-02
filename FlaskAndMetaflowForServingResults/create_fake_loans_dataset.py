"""

Simple stand-alone script to create a txt file representing a fake Loan approval dataset 
to train a classification model later on.

"""

import numpy as np
import pandas as pd


def main():
    n_samples = 1000
    
    zipcodes = np.random.randint(10000, 99950, n_samples)
    df = pd.DataFrame({"zipcode":zipcodes})
    df['age'] = np.random.randint(21, 65, n_samples)
    df['race'] = np.random.choice(["Asian","White", "African American","American Indian","Hispanic"], size=n_samples, p=[.05,.60,.20,.05,.10])
    df['gender'] = np.random.choice(["Male","Female","Other"], size=n_samples, p=[.33,.33,.34])
    df['customer_type'] = np.random.choice(["new","old"], size=n_samples, p=[.30,.70])
    df['credit_score'] = np.random.randint(600, 850, n_samples)
    df['loan_amount'] = np.random.randint(500, 10000, n_samples)
    df['loan_approved'] = np.random.choice([1, 0], size=n_samples, p=[.50,.50]) 
    
    df.to_csv('loans_dataset.csv', encoding='utf-8', index=False)


if __name__ == "__main__":
    main()