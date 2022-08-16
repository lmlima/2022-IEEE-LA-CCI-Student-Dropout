# Student Dropout Prediction using 1D CNN-LSTM with Variational Autoencoder Oversampling

Official implementation of our 2022 IEEE LA-CCI paper Student Dropout Prediction using 1D CNN-LSTM with Variational Autoencoder Oversampling by Eduarda C. Coppo, Rhuan S. Caetano, Leandro M. de Lima and Renato A. Krohling.

## Install

`conda install numpy pandas matplotlib tensorflow==2.2.0 plotly tabulate scikit-learn seaborn keras ipython jupyter`

`pip install optuna`

## Enviroment config
`conda create --name <env> --file env.yml`

## Autoencoer config
### In file src\imbalanced-sequence-classification-master\utils\config.py
### TIMESTEPS: sequence lenght
`TIMESTEPS = 4`

### DATA_DIM: number of features
`DATA_DIM = 26`

### NUM_CLASSES: number of classes
`NUM_CLASSES = 2`

## Acknowledgement
We borrowed and modified code from ["Autoencoders and Generative Adversarial Networks for Anomaly Detection for Sequences"](https://github.com/stephanieger/imbalanced-sequence-classification) by Stephanie Ger and Diego Klabjan. We would like to expresse our gratitdue for the authors of these repositeries.
