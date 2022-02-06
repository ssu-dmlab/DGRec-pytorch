
# DGRec-pytorch

This repository implements **DGRec** proposed in "Session-Based Social Recommendation via Dynamic Graph Attention Network (WSDM 2019)" using PyTorch. 
We refer to the original code [(link)](https://github.com/DeepGraphLearning/RecommenderSystems/tree/master/socialRec) which is implemented by Tensorflow.

## Getting Started

### Installation

This project requires Python 3.8 and the following Python libraries:
- numpy == 1.22.0
- pandas == 1.3.5
- matplotlib == 3.5.1
- torch == 1.8.2
- fire == 0.4.0
- tqdm == 4.62.3
- loguru == 0.5.3
- dotmap == 1.3.17

To install the pacakges used in this repository, type the following command at the project root:
```bash
pip install -r requirements.txt
```

### Data Preparation

Before running the model, you need to prepare datasets to be trained as follows:
```bash
unzip datasets/$DATASET.zip -d datasets/$DATASET/`
```
where `$DATASET` is one of `bookdata`, `musicdata` and `moviedata`. 


### How To Run
You can simply check if the model works correctly by typing the following command:
```
PYTHONPATH=src python3 run.py --data_name $DATASET
```
The above command will start learning the model on the `$DATASET` with the specified parameters saved in `param.json`.

## Usage
To use those scripts properly, move your working directory to `./src`.

You can tune the hyperparameters and run this project to simply type the following in your terminal:

```
python3 -m main_trainer \
        --model ··· \
        --data_name ··· \
        --seed ··· \
        --epochs ··· \
        --act ··· \
        --batch_size ··· \
        --learning_rate ··· \
        --embedding_size ··· \
        --max_length ··· \
        --samples_1 ··· \
        --samples_2 ··· \
        --dropout ··· \
```
  
### Arguments

|Arguments|Explanation|Default|
|------|---|---|
|model|Name of model|'DGRec'|
|data_name|Name of data|'bookdata'|
|seed|Random seed|0|
|epochs|Number of traning epochs |20|
|act|Type of activation function|'relu'|
|batch_size|Size of batch|100|
|learning_rate|Learning rate of model|0.002|
|embedding_size|Size of item and user embedding|100|
|max_length|Max item count reflected in each user_embedding|20|
|samples_1|Number of target user's friends|10|
|samples_2|Number of target user's friends' friends|5|
|dropout|Dropout rate|0.2|

## Repository Structure

The overall file structure of this repository is as follows:

```
DGRec-pytorch
    ├── README.md
    ├── datasets                        
    ├── requirments
    ├── run.py                          # simply run model with hyperparameter which already specified                 
    └── src         
        ├── utils.py                    # script for setting random seed and showing hyperparameters
        ├── data.py                     # In charge of data loading 
        ├── main.py                     # input processing for model training
        └── models                      
            ├── __init__.py
            ├── eval.py                 # evaluate model with validation data
            ├── model.py                # model implementation (forward)
            ├── train.py                # model training with data and hyperparameter
            └── batch
                ├── __init__.py
                ├── minibatch.py        # split and set data to minibatch
                └── neigh_sampler.py    # Classes that are used to sample node neighborhoods
```

## Data

### Input Data Files
* `train.tsv`: includes user historical behaviors, which is organized by pandas.Dataframe in five fields (SessionId UserId ItemId Timestamps TimeId).
* `valid.tsv`: the same format as train.tsv, used for tuning hyperparameters.
* `test.tsv`: the same format as test.tsv, used for testing model.
* `adj.tsv`: includes links between users, which is also organized by pandas.Dataframe in two fields (FromId, ToId).
* `latest_session.tsv`: serves as 'reference' to target user. This file records all users available session at each time slot. For example, at time slot t, it stores user u's t-1 th session.
* `user_id_map.tsv`: maps original string user id to int.
* `item_id_map.tsv`: maps original string item id to int.

### Data Statistics
The statistics of the Douban datasets, `bookdata` and `musicdata`, are summarized as follows:

|Dataset|#user|#item|#event|
|------|---|---|---|
|`bookdata`|46,548|212,995|1,908,081|
|`musicdata`|39,742|164,223|1,792,501|
|`moviedata`|94,890|81,906|11,742,260|

## Experiments

We have test the accuracy compared to the original result. We report average metrics with thier standard deviations of 10 runs.

* [ ] Original version - tensorflow

|data|recall@20|ndcg|
|------|---|---|
|`bookdata`|0.3771 |0.2841|
|`musicdata`|0.3382|0.2539|
|`moviedata`|0.1861|0.1950|

* [ ] Implemented version - pytorch

|data|recall@20|ndcg|
|------|---|---|
|book data ± (std deviation)|0.3673 ± (0.0261)|0.3035 ± (0.0130)|
|music data ± (std deviation)|0.3529 ± (0.0295)|0.2962 ± (0.0103)|
|movie data ± (std deviation)|0.1594 ± (0.0031)|0.1955 ± (0.0015)|


 ## References
 [1] Song, W., Xiao, Z., Wang, Y., Charlin, L., Zhang, M., & Tang, J. (2019, January). Session-based social recommendation via dynamic graph attention networks. In Proceedings of the Twelfth ACM international conference on web search and data mining (pp. 555-563).
