# Hypertriton3Body

Contains the code and the data to study the hypertriton production in Pb-Pb collisions at 5.02TeV via its three body decay channel. Analysis based on xgboost Boosted Decision Trees Classifier.

## Run the analysis
- Download the data and set the environment: `python config.py` 
- Generate flat trees for feeding the BDT: `cd GenerateTables/`, then `python table_generator.py`
- Train and test the BDT model: `cd Analysis/`, then `python train_and_apply`. The file `Config.yaml` allows you to customize the training and define a BDT      efficiency range for computing the systematic variations
- Extract the hypertriton signal and compute the lifetime: `python lifetime_3body.py`
- Check the results: `cd Results/`
