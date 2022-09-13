

python run_training_mlp.py --features CP DT SUT SUT_DT_ADD SW_DW_ADD50 --model_name mlp_5_features

python run_training_mlp.py --features ^
CP DT SUT SUT_DT_ADD SW_DW_ADD10 SW_DW_ADD25 SW_DW_ADD33 SW_DW_ADD50 SW_DW_ADD66 SW_DW_ADD75 DW10 SW_DW_DIV_50 SW_DW_DIV_75 SW_DW_DIV_33 SW_DW_DIV_10^
 -n mlp_15_features



python run_training_mlp.py --features ^
 CP DT SUT SUT_DT_ADD SW_DW_ADD10 SW_DW_ADD25 SW_DW_ADD33 SW_DW_ADD50  SW_DW_ADD66 SW_DW_ADD75 DW10 SW_DW_DIV_50 ^
 -n mlp_12_features


python run_training_mlp.py --features ^
CP DT SUT SUT_DT_ADD ^
 -n mlp_4_features



python run_training_mlp.py -n mlp_21_features

