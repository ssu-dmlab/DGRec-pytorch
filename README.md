
# DGRec-pytorch

Pytorch implementation model of 'Session-Based Social Recommendation via Dynamic Graph Attention Network' which is implemented by Tensorflow.
Template code is provided at the https://github.com/DeepGraphLearning/RecommenderSystems/tree/master/socialRec which is original version code implemented by tensorflow

## Getting Started

### Installation

This project requires Python 3.8 and the following Python libraries installed:
- Numpy = 1.22.0
- Pandas = 1.3.5
- matplotlib = 3.5.1
- torch = 1.8.2+cu111
- fire = 0.4.0
- tqdm = 4.62.3
- loguru = 0.5.3


### How to run (terminal)
In a terminal or command window, navigate to the top-level project directory `DGRec-pytorch/` (that contain this README) and run the following commands:

1. `pip install -r requirements.txt`
2. `unzip datasets/$DATASET.zip -d datasets/$DATASET/` where `$DATASET` is `bookdata` or `musicdata`
3. `cd src`
4. `python3 -m main`

This will run the model

### Usage

In a terminal or command window, argument can be apply by typing

    python3 -m main \
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

  
### Arguments

|Arguments|Explanation|Default|
|------|---|---|
|model|name of model|'DGRec'|
|data_name|name of data|'bookdata'|
|seed|seed number|0|
|epochs|# of train |40|
|act|activation function|'relu'|
|batch_size|size of batch|100|
|learning_rate|learning rate of model|0.002|
|embedding_size|size of item and user embedding|100|
|max_length|max item count reflected in each user_embedding|20|
|samples_1|number of target user's friends|10|
|samples_2|number of target user's friends' friends|5|
|dropout|dropout rate|0.2|

## Repository Overview

Provide an overview of the directory structure and files:

    ├── README.md
    ├── datasets
    ├── requirments
    └── src
        ├── utils.py
        ├── data.py
        ├── main.py
        └── models
            ├── __init__.py
            ├── eval.py
            ├── model.py
            ├── train.py
            └── batch
                ├── __init__.py
                ├── minibatch.py
                └── neigh_sampler.py

## Data

### Input data:
* `train.tsv`: includes user historical behaviors, which is organized by pandas.Dataframe in five fields (SessionId UserId ItemId Timestamps TimeId).
* `valid.tsv`: the same format as train.tsv, used for tuning hyperparameters.
* `test.tsv`: the same format as test.tsv, used for testing model.
* `adj.tsv`: includes links between users, which is also organized by pandas.Dataframe in two fields (FromId, ToId).
* `latest_session.tsv`: serves as 'reference' to target user. This file records all users available session at each time slot. For example, at time slot t, it stores user u's t-1 th session.
* `user_id_map.tsv`: maps original string user id to int.
* `item_id_map.tsv`: maps original string item id to int.

### Raw data:
The statistics of `Douban datasets` are summarized as follows:

|Dataset|#user|#item|#event|
|------|---|---|---|
|DoubanMusic|39,742|164,223|1,792,501|
|DoubanBook|46,548|212,995|1.908.081|

## Experiments

We have test the accuracy compared to the original result. We report average metrics with thier standard deviations of 10 runs.

* [ ] Original version - tensorflow

|data|recall@20|ndcg|
|------|---|---|
|book data|0.3771 |0.2841|
|music data|0.3382|0.2539|

* [ ] Implemented version - pytorch

|data|recall@20|ndcg|
|------|---|---|
|book data ± (std deviation)|0.3619 ± (0.0216)|0.2911 ± (0.0059)|
|music data ± (std deviation)|0.3431 ± (0.0225)|0.2939 ± (0.0095)|


 ## References
 [1] Song, W., Xiao, Z., Wang, Y., Charlin, L., Zhang, M., & Tang, J. (2019, January). Session-based social recommendation via dynamic graph attention networks. In Proceedings of the Twelfth ACM international conference on web search and data mining (pp. 555-563).
