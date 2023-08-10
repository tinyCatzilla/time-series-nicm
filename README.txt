############### JOBS ###############

Navigate to ./jobs/, which contains all SLURM batch jobs.
Running any of cmr.jobs, echo.jobs, no_locf.jobs will preprocess the data, train and evaluate all models, create latex tables of the results, and store the results in ./results
More indepth explanation is below, but basically you can just run these and after the script is finished, navigate to ./results/last_cmr, ./results/last_echo, or ./results/last_no_locf respectively.

############### FILE STRUCTURE ###############

data:
    cmr:
        dx_processed - Feature data using CMR index events and LOCF.
        labels_processed - Ground truths with dates of CMR index events.
        data_train, data_test, data_val - dx_processed split into respective sets.
    echo:
        same as cmr, but using echo index events.
    no_locf:
        same as cmr, but without LOCF performed
    split:
        test_pids, train_pids, val_pids - csv files containing the patient IDs of each CMR dataset

jobs:
    data.jobs:
        computes descriptions of the datasets using utils/describe_data.R 
    cmr.jobs:
        1. processes the data with cmr events and LOCF using prep/prep_data_cmr.R
        2. runs models/cmr.py, the main script for model training and evaluation
        3. Runs utils/latex.R and utils/median.R to generate latex tables from the results.
        4. Renames the output to be stored at results/last_cmr
    echo.jobs:
        Same as cmr, but first runs prep/proc_echo.R, a script that only needs be run once but is quick enough to not matter.
        Also, output is in results/last_echo
    no_locf.jobs:
        Same as cmr, but output is in results/last_no_locf

models:
    cmr.py, echo.py, no_locf.py:
        Identical in all but hyperparameters and in the case of echo, feature size (due to rare observation pruning).
        Well documented but in summary works as follows:
            1. Instantiate datasets and models
            2. Train one model
            3. Calibrate probabilities and fit isotonic regression on train set
            4. Evaluate on validation/test set
            5. Repeat 2-4 until all models are complete
            6. Evaluate 'ensemble' model
    m_dataset.py:
        custom dataloader helper function, just to ensure we properly associate patient time-series EHRs with ground truths

results:
    base:
        contains output while models are currently running.
        structured with folder for each model, and within each folder there is
            - calibration_curve_grid.png - shows a rough graph of what the isotonic regression and probability calibration did for the evaluation set
            - eval_calibrated.csv - model evaluation post-calibration and isotonic regression
            - eval.csv - model evaluation with no techniques applied
            - logs.csv - logs from each epoch
    last_cmr, last_echo, last_no_locf:
        Same as base, but for the (respective) last completed .jobs batch.
    test_cmr, test_echo, test_no_locf, test_weighted:
        test results currently (8/10) being used in the paper
    
utils:
    describe_data.R - computes descriptions of the datasets, formatted as a latex table
    latex.R - aggregates model results, formatted as a latex table
    median.R - aggregates median model results for each class, formatted as a latex table
    pull_data_sql.py - script for original SQL pull
    split.py - script for original train/val/test split, including the random seed used.

zOut:
    output files for each script