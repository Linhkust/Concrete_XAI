# Concrete_XAI
Codes for the paper "The effects of recycled brick aggregates and basalt fiber on the compressive strength of recycled aggregate concrete: experiment and explainable artificial intelligence methods"

### FOR USERS
The web GUI called Concrete_XAI has been deployed and is free to use by clicking the [LINK](https://civil3591.shinyapps.io/concretexai/).

### FOR DEVELOPERS
For developers who want to customize the app, the instructions have been listed as follows.
#### Step 1: Create the virtual Python environment with the following command
`conda create -n my_env_name python=3.9`

#### Step 2: Activate the virtual environment with the following command
`conda activate my_env_name`

#### Step 3: Install all necessary dependencies listed in the `requirements.txt` with the following command:
`pip install requirements.txt`

#### Step 4: Run `app.py` and click the link [http://127.0.0.1:8000](http://127.0.0.1:8000)
The web GUI will show on the browser and is ready for use.

#### Module 1: Model Generation and Evaluation
![Module 1](https://github.com/Linhkust/Concrete_XAI/blob/main/web%20gui%20screenshots/1.png)

#### Module 2: Explainable Artificial Intelligence
![Module 2: PFI](https://github.com/Linhkust/Concrete_XAI/blob/main/web%20gui%20screenshots/2-1.png)
![Module 2: SHAP Summary](https://github.com/Linhkust/Concrete_XAI/blob/main/web%20gui%20screenshots/2-2.png)
![Module 2: PDP and SHAP dependence plot](https://github.com/Linhkust/Concrete_XAI/blob/main/web%20gui%20screenshots/2-3.png)

#### Module 3: Large language Model Analysis
![Module 3](https://github.com/Linhkust/Concrete_XAI/blob/main/web%20gui%20screenshots/3.png)
