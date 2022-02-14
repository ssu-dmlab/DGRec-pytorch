
# DGRec-pytorch

This repository implements **DGRec** proposed in "Session-Based Social Recommendation via Dynamic Graph Attention Network (WSDM 2019)" using PyTorch. 
We refer to the original code [(link)](https://github.com/DeepGraphLearning/RecommenderSystems/tree/master/socialRec) which was implemented by Tensorflow.

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
You can simply check if the model works correctly with the following command:
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
    ├── run.py                          # starts training the model with specified hyperparameters
    └── src         
        ├── utils.py                    # contains utility functions such as setting random seed and showing hyperparameters
        ├── data.py                     # loads a specified dataset
        ├── main.py                     # processes input arguments of a user for training
        └── models                      
            ├── __init__.py
            ├── eval.py                 # evaluates the model with a validation dataset
            ├── model.py                # implements the forward function of the model
            ├── train.py                # implements a function for training the model with hyperparameters
            └── batch
                ├── __init__.py
                ├── minibatch.py        # splits a dataset to mini-batches
                └── neigh_sampler.py    # samples neighborhoods of a given node
```

## Data

### Input Data Files
The datasets are from the original repository [(link)](https://github.com/DeepGraphLearning/RecommenderSystems/tree/master/socialRec). The format of each file is as follows:

* `train.tsv`: includes historical behaviors of users. The data in the file is loaded into pandas.Dataframe with five fields such as (SessionId UserId ItemId Timestamps TimeId)
* `valid.tsv`: the same format as `train.tsv`, used for tuning hyperparameters.
* `test.tsv`: the same format as `train.tsv`, used for testing model.
* `adj.tsv`: an edge list representing relationships between users, which is also organized by pandas.Dataframe in two fields (FromId, ToId).
* `latest_session.tsv`: serves as 'reference' to target user. This file records all users available session at each time slot. For example, at time slot t, it stores user u's t-1 th session.
* `user_id_map.tsv`: maps original string user name to integer id.
* `item_id_map.tsv`: maps original string item name to integer id.

### Data Statistics
The statistics of the Douban datasets, `bookdata` and `musicdata`, are summarized as follows:

|Dataset|#user|#item|#event|
|------|---|---|---|
|`bookdata`|46,548|212,995|1,908,081|
|`musicdata`|39,742|164,223|1,792,501|
|`moviedata`|94,890|81,906|11,742,260|

## Experiments

We compare our implmentation compared to the original code in terms of recall@20 and ndcg. We report average metrics with thier standard deviations of 10 runs.

* Original version based on Tensorflow

|data|recall@20|ndcg|
|------|---|---|
|`bookdata`|0.3771 |0.2841|
|`musicdata`|0.3382|0.2539|
|`moviedata`|0.1861|0.1950|

* Our version based on PyTorch

|data|recall@20|ndcg|
|------|---|---|
|book data ± (std deviation)|0.3718 ± (0.0250)|0.3066 ± (0.0116)|
|music data ± (std deviation)|0.3529 ± (0.0295)|0.2962 ± (0.0103)|
|movie data ± (std deviation)|0.1594 ± (0.0031)|0.1955 ± (0.0015)|


## References
[1] Song, W., Xiao, Z., Wang, Y., Charlin, L., Zhang, M., & Tang, J. (2019, January). Session-based social recommendation via dynamic graph attention networks. In Proceedings of the Twelfth ACM international conference on web search and data mining (pp. 555-563).
