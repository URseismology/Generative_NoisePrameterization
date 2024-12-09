This readme file is created to explain the folder structure and instrcution related to execution of the diffusion model.

--------------------------------------------------------------------------------------
Folder Structure
--------------------------------------------------------------------------------------

Root Folder: PRJ_AUTO_NOISEPARAM
Subfolders:
    1. code
        a.analysis: contains general data analysis, data processing and results analysis jupyter notebooks.
        b.lib: contains the utility codes, the diffusion model related classes and methods.
        c.ipynb_single_run_code_files: contains all methods/function definitions and model run in one place as a single excutable jupyter notebook.The files are named as per expriments mentioned in the report.

    2. data
        a.processed: contains the processed pickle/data files ready for model input
        b.raw: raw seismic data. currently empty. not kept due to huge size.

    3: models: contains saved models from multiple runs

    4. results: contains figures and results of the generative model


--------------------------------------------------------------------------------------
Model Execution Details
--------------------------------------------------------------------------------------

    > train model: to train model with optimal hyperparameters execute, "train_diff_model_final.py"
    
    > model optimization: for hyperparameter optimization execute "train_diff_model_wandb_hyperparam_opt.py"
    

--------------------------------------------------------------------------------------
Result Analysis Details
--------------------------------------------------------------------------------------

    > to analyze the results of the generative model execute "code/analysis/results_analysis.ipynb" jupyter notebook
