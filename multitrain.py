import domino
import os
import time

# In this project we will use Domino's platform API through it's Python wrapper to kick off several training jobs at the same time!

print('Initializing Domino Project for API calls')

# Initialize Domino Project
# Domino automatically passes down environment variables about the project, owner and even our individual API key and Token file for executing the Jobs in Domino.
domino_project =domino.Domino(project = str(os.environ.get('DOMINO_PROJECT_OWNER')+'/'+os.environ.get('DOMINO_PROJECT_NAME')),
                              api_key = os.environ.get('DOMINO_USER_API_KEY'),
                              domino_token_file=os.environ.get('DOMINO_TOKEN_FILE'))


# h2o_model_train
print('Kicking off sklearn model training')
domino_project.job_start(command='/mnt/code/scripts/h2o_model_train.py')

# sklearn_log_reg_train
print('Kicking off R model training')
domino_project.job_start(command='/mnt/code/scripts/sklearn_log_reg_train.py')

# sklearn_RF_train
print('Kicking off h2o model training')
domino_project.job_start(command='/mnt/code/scripts/sklearn_RF_train.py')

# xgb_model_train
domino_project.job_start(command='/mnt/code/scripts/xgb_model_train.py')

print('Done!')





