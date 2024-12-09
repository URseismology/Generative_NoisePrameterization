This readme file is created to explain the folder structure and instruction related to execution of the diffusion model.

--------------------------------------------------------------------------------------
Abstract
--------------------------------------------------------------------------------------

The receiver function (RF) technique is a way to study the structure of the Earth’s crust and upper mantle by using data from teleseismic earthquakes. When seis- mic signals are recorded they are always convolved with inter crustal reverbera- tions and noise. These makes the singal anaysis harder and in order to analyze Seismologist needs to deconvolve those noises to generate accurate receiver func- tions which can then be used to analyse velocity structure beneath the receiver. Though a variety of deconvolution techniques have been developed, they are all adversely affected by background and signal-generated noise. In order to take into account the unknown noise characteristics Seismologist has previously used Bayesian inference in which both the noise magnitude and noise spectral charac- ter are parameters in calculating the receiver functions. In this project I propose to build a generative deep learning diffusion model which can better estimate the noise information and thus in turn can imporve a more accurate receiver function realization.

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
        a.processed: contains the processed pickle/data files ready for model input. currently empty. not kept due to huge size.
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
