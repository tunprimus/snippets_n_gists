#!/usr/bin/env python3
# Adapted from Synthetic Data Generation: A Hands-On Guide in Python -> https://www.datacamp.com/tutorial/synthetic-data-generation
import numpy as np
import random
import simpy
try:
    import fireducks.pandas as pd
except ImportError:
    import pandas as pd


def synthesise_height_weight_data(num_samples=1000):
    """
    Synthesise height and weight data based on a normal distribution.

    The height variable is sampled from a normal distribution with a mean of 173 and a standard deviation of 13.
    The weight variable is sampled from a normal distribution with a mean of 67 and a standard deviation of 17.
    The height and weight variables are then correlated so that for every 1cm increase in height, the weight increases by 0.53kg.

    Parameters
    ----------
    num_samples : int
        The number of samples to generate.

    Returns
    -------
    df : pd.DataFrame
        A DataFrame containing the synthesised data with the following columns:
        * gender : str (either "male" or "female")
        * height : float
        * weight : float
    """
    import numpy as np
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd

    rng = np.random.default_rng()
    heights = rng.normal(loc=173, scale=13, size=num_samples)
    weights = rng.normal(loc=67, scale=17, size=num_samples)
    # Create correlation between height and weight
    weights += (heights - 173) * 0.53

    df = pd.DataFrame({"height": heights, "weight": weights})
    # Add categorical data
    genders = rng.choice(["male", "female"], size=num_samples)
    df["gender"] = genders
    df = df[["gender", "height", "weight"]]
    return df


def synthesise_human_profile_data(num_records=23):
    """
    Generate synthetic human profile data using the Faker library.

    This function creates a DataFrame with randomly generated human profile data,
    including attributes like name, date of birth, phone number, address, email,
    country, company name, and job title.

    Parameters
    ----------
    num_records : int, optional
        The number of human profile records to generate. Default is 23.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the synthetic human profile data with the following columns:
        - 'name': str
        - 'date_of_birth': datetime.date
        - 'phone': str
        - 'address': str
        - 'email': str
        - 'country': str
        - 'company_name': str
        - 'job_title': str
    """
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    from faker import Faker

    # Initialise the Faker instance
    fake = Faker()
    data = []
    for _ in range(num_records):
        data.append({
            "name": fake.name(),
            "date_of_birth": fake.date_between("-67y", "-18y"),
            "phone": fake.phone_number(),
            "address": fake.address(),
            "email": fake.email(),
            "country": fake.country(),
            "company_name": fake.company(),
            "job_title": fake.job(),
            })
    df_fake = pd.DataFrame(data)
    return df_fake


def synthesise_customer_loan_data(num_records=1000):
    """
    Synthesise customer loan data based on age, income, and credit score.

    The rules used to generate the synthetic data are as follows:
    Rule 1: Income is loosely based on age
    Rule 2: Credit score is influenced by age and income
    Rule 3: Loan amount is based on income and credit score

    Parameters
    ----------
    num_records : int, optional
        The number of customer loan records to generate. Default is 1000.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the synthetic customer loan data with the following columns:
        - 'age': int
        - 'income': float
        - 'credit_score': int
        - 'loan_amount': float
    """
    import numpy as np
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd

    rng = np.random.default_rng()
    data = []
    for _ in range(num_records):
        age = rng.integers(18, 84)

        # Rule 1: Income is loosely based on age
        base_income = 20000 + (age - 18) * 997
        income = rng.normal(loc=base_income, scale=base_income * 0.19)

        # Rule 2: Credit score is influenced by age and income
        credit_score = min(850, max(300, int(600 + (age/84) * 100 + (income/100000) * 100 + rng.normal(0, 53))))

        # Rule 3: Loan amount is based on income and credit score
        max_loan = income * (credit_score / 600)
        loan_amount = rng.uniform(0, max_loan)

        data.append([age, income, credit_score, loan_amount])

    df = pd.DataFrame(data, columns=["age", "income", "credit_score", "loan_amount"])
    return df


class Bank(object):
    def __init__(self, env, num_tellers):
        self.env = env
        self.tellers = simpy.Resource(env, num_tellers)

    def service(self, customer):
        service_time = random.expovariate(1/10)
        yield self.env.timeout(service_time)


def customer(env, name, bank, data):
    """
    Simulates a customer arriving at the bank and waiting for service.

    Parameters
    ----------
    env : simpy.Environment
        The simulation environment
    name : str
        The customer's name
    bank : Bank
        The bank object
    data : list
        A list to store the customer's data

    Yields
    ------
    simpy.events.Process
        A process that represents the customer's service time
    """
    arrival_time = env.now
    print(f"{name} arrives at the bank at {arrival_time:.2f}")
    with bank.tellers.request() as request:
        yield request
        wait_time = env.now - arrival_time
        print(f"{name} waits {wait_time:.2f} for a teller")

        yield env.process(bank.service(name))

        service_time = env.now - arrival_time
        print(f"{name} leaves the bank at {env.now:.2f}")
        data.append((name, arrival_time, wait_time, service_time))

def run_simulation(env, num_customers, bank, data):
    """
    Simulate the arrival, waiting, and service of customers in a bank environment.

    This function schedules customer processes in a simulation environment. Each customer arrives
    at the bank, waits for service if necessary, and is then serviced by one of the bank tellers.
    The arrival times are generated based on an exponential distribution.

    Parameters
    ----------
    env : simpy.Environment
        The simulation environment in which the processes run.
    num_customers : int
        The total number of customers to simulate.
    bank : Bank
        The bank object that manages the tellers and service.
    data : list
        A list to store each customer's data, including name, arrival time, wait time, and service time.
    """
    import random

    for i in range(num_customers):
        env.process(customer(env, f"Customer {i}", bank, data))
        yield env.timeout(random.expovariate(1/5))


# Set up and run the simulation
def synthesise_bank_service_data(num_customers=103, num_tellers=3):
    """
    Simulate the arrival, waiting, and service of customers in a bank environment.

    Parameters
    ----------
    num_customers : int, optional
        The total number of customers to simulate. Default is 103.
    num_tellers : int, optional
        The number of tellers in the bank. Default is 5.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the synthetic customer service data with the following columns:
        - 'customer_name': str
        - 'arrival_time': float
        - 'wait_time': float
        - 'service_time': float
    """
    import random
    import simpy
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    env = simpy.Environment()
    bank = Bank(env, num_tellers)
    data = []
    env.process(run_simulation(env, num_customers, bank, data))
    env.run()
    # Create a DataFrame from the collected data
    df = pd.DataFrame(data, columns=["customer_name", "arrival_time", "wait_time", "service_time"])
    return df


synthesise_height_weight_data()
synthesise_human_profile_data()
synthesise_customer_loan_data()
synthesise_bank_service_data()
