# Neural-Based Blood Pressure estimation


# Summary
This project is an implementation of two neural models for neural blood pressure estimation from Photoplethysmography(PPG) data.
The first model is Convolutional Neural network combined to a bidirectional LSTM and the second model is a Multi Layer perceptron.
The current code is written using pytorch and  includes both data preprocessing and models' training and testing.


# Data Preprocessing.

## Requirements.
The raw PPG data from MMIC dataset directory "Rec_mmic" must be pasted at the root directory of the project


## Extracting periodic segments for CNN+BILSTM model
To generates segments from periods of PPG signal, run the following code.
A sampling of periods is done so that with obtain above 50000
segments.
```
python .\preprocessing\prepare_windows_dataset.py
```
The data extracted are stored in csv files in directories.
data/windows_data_mmic/train|test|val


 
## Extracting temporal segments for CNN+BILSTM model
To generates segments from temporal windows of 5.6 secondes (700 points.) of PPG signal, run the following code.
A sampling of periods is done so that with obtain above 50000
segments.

```
python .\preprocessing\prepare_temporal_windows_dataset.py
```
The data extracted are stored in csv files in directories.
data/temporal_windows_data_mmic/train|test|val




## Extracting features segments for MLP model
To extract 21 features for PPG signal . Segments are sampled to obtain
above 15000 samples.
```
python .\preprocessing\prepare_features_mlp_dataset.py
```
The data extracted are stored in csv files in directories.
data/features_data_mmic/train|test|val



# Networks 

## MLP 
The MLP is defined in  ./networks/mlp.py

## CNN+BiLSTM
The CNN model is defined in ./networks/cnnlstm.py



# Training 
Both models are trained using the same Trainer class defined in 
./training/trainer.py


## Training the CNN with windows data.
```
python run_training_cnn.py --model_name [MODEL_NAME] --epochs [NUMBER_EPOCHS]  --lr [LEARNING_RATE]
```
Example  
```
python run_training_cnn.py --model_name MY_MODEL --epochs 50  --lr 0.0001

```
The model will train for 50 epochs and the results with be stored in logs/experiments/MY_MODEL
: tensorboard curves, test_results.csv and learning_curve.csv


## Training the CNN with windows data.
To run with temporal data, just perform a slight modification in ./constants.py
by setting the value of  variable WINDOWS_DATA_DIR to "os.path.join(ROOT_DIR,"data/temporal_windows_data_mmic") 



## Training the MLP 
```
python run_training_mlp.py --model_name [MODEL_NAME] --epochs [NUMBER_EPOCHS]  --lr [LEARNING_RATE]
--features [FEATURES]
```
--features parameters is the list of features to use. By default it set to all 21 extracted features.

Example :
```
python run_training_mlp.py --features CP DT SUT SUT_DT_ADD SW_DW_ADD50 --model_name mlp_5_features 
```
to train the MLP using the 5 features CP, DT, SUT, SUT_DT_ADD and SW_DW_ADD50. 
Similarly to the CNN training. The results are store in ./logs/experiments/MODEL_NAME.


