# Run training
1. Copy all the datasets subfolders (i.e. babylm_100M, babylm_10M, babylm_dev, and babylm_test) and put them in a new folder 'datasets' (i.e. baseline-pretraining-main/datasets/babylm_10M)
2. Go to folder 'pt_framework-master' and run 'pip install .' 

3a. Navigate to the datasets\babylm_10M folder in Bash
3b. Run cat *.train > babylm_10M.txt
3c. Navigate to the datasets\babylm_100M folder in Bash
3d. Run cat *.train > babylm_100M.txt
3e. Navigate to the datasets\babylm_dev folder in Bash
3f. Run cat *.dev > babylm_dev.txt
3g. Navigate to datasets\babylm_test folder in Bash
3h. Run cat *.test > babylm_test.txt

4. You can now delete the rest of the .train/.test/.dev files if you want.
5. Change line 9 and 10 in scripts\train_t5_babylm.sh to your own location
6. Run ./train_t5_babylm.sh via Bash in folder scripts

# Where important parameters are defined:
Learning rate schedule is defined at function 'get_learning_rate_params' in script 'basic_param_setter.py' under src/babylm_baseline_train folder.

# How to load the pretrained models:
See the functions in src/babylm_baseline_train/models/ckpt_loader.py.