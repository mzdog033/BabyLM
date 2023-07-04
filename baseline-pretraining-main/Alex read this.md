# Run training
1. Put the babylm_100M dataset in datasets\babylm_100M
2. Navigate to the datasets\babylm_100M folder in Bash
3. Run cat *.train > babylm_100M.txt
4. You can now delete the rest of the .train files if you want.
5. Go to folder 'pt_framework-master' and run 'pip install .' 
6. Change line 9 and 10 in scripts\train_t5_babylm.sh to your own location
7. Run ./train_t5_babylm.sh via Bash in folder scripts

# Where important parameters are defined:
Learning rate schedule is defined at function 'get_learning_rate_params' in script 'basic_param_setter.py' under src/babylm_baseline_train folder.

# How to load the pretrained models:
See the functions in src/babylm_baseline_train/models/ckpt_loader.py.