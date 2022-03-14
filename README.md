
# DGRec-pytorch

This repository implements **DGRec** proposed in "Session-Based Social Recommendation via Dynamic Graph Attention Network (WSDM 2019)" using PyTorch. 
We refer to the original code [(link)](https://github.com/DeepGraphLearning/RecommenderSystems/tree/master/socialRec) which was implemented in Tensorflow.

This repository is developed by Taesoo Kim, and commented by [Jinhong Jung](https://jinhongjung.github.io/).

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
unzip datasets/$DATASET.zip -d datasets/$DATASET/
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
        --decay_rate ... \
        --gpu_id ... \
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
|embedding_size|Size of item and user embedding|50|
|max_length|Max item count reflected in each user_embedding|20|
|samples_1|Number of target user's friends|10|
|samples_2|Number of target user's friends' friends|5|
|dropout|Dropout rate|0.2|
|decay_rate|weight decay rate|0.98|
|gpu_id|Id of gpu you use|0|

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
The statistics of `bookdata`, `musicdata`, and `moviedata` from the Douban domain are summarized as follows:

|Dataset|#user|#item|#event|
|------|---|---|---|
|`bookdata`|46,548|212,995|1,908,081|
|`musicdata`|39,742|164,223|1,792,501|
|`moviedata`|94,890|81,906|11,742,260|

## Experiments

We compare our implmentation compared to the original one in terms of recall@20 and ndcg. We report average metrics with thier standard deviations of 10 runs.

### Recommendation accuracies on test sets using the original version in Tensorflow

|data|recall@20|ndcg|
|------|---|---|
|`bookdata`|0.3771 |0.2841|
|`musicdata`|0.3382|0.2539|
|`moviedata`|0.1861|0.1950|


### Recommendation accuracies on test sets using our version in Pytorch

|data|recall@20|ndcg|
|------|---|---|
|`bookdata`|0.4034 ± (0.0160)|0.3140 ± (0.0059)|
|`musicdata`|0.3777 ± (0.0164)|0.2947 ± (0.0113)|
|`moviedata` |0.1594 ± (0.0031)|0.1955 ± (0.0015)|

* Note that the above results were obatined with the early-stopping technqiue w.r.t. ndcg, i.e., if you change the base metric of the early-stopping to recall@20, the test recall of our version can be improved as similar as that of the original version. 

### Hyperparameter tuning

The following tables summarize the experimental results to find better hyperparameters for each dataset.

#### Results on `bookdata` (batch_size : 100)
We first fixed `batch_size` to 100, and varied the size of embeddings as follows:

|Embedding size|recall@20|ndcg|
|--------------|---------|----|
|100|0.3673 ± (0.0261)|0.3035 ± (0.0130)|
|50|0.3718 ± (0.0250)|0.3066 ± (0.0116)|
|30|0.3551 ± (0.0431)|0.2998 ± (0.0132)|

The table shows that when the embedding size is 50, the model produces the best accuracy among the results. 
Then, we fixed the embedding size to 50, and checked the effects of activation functions, drop-out, learning rate, and decay ratio as follows:

|Activation|Drop-out|Learning rate|Decay ratio|recall@20|ndcg|
|--------------|---------|----|----|----|----|
|relu|0.2|0.002|0.99|0.3449 ± (0.0181)|0.3002 ± (0.0091)|
|relu|0.2|0.002|0.99|0.2904 ± (0.0172)|0.2705 ± (0.0092)|
|relu|0.2|0.002|0.98|0.1894 ± (0.0233)|0.2239 ± (0.0039)|
|relu|0.2|0.01|0.95|0.3877 ± (0.0341)|0.3106 ± (0.0083)|
|relu|0.2|0.01|0.99|0.3911 ± (0.0271)|0.3166 ± (0.0091)|
|relu|0.2|0.01|0.98|0.3705 ± (0.0219)|0.3089 ± (0.0090)|
|relu|0.3|0.002|0.95|0.3391 ± (0.0193)|0.2999 ± (0.0071)|
|relu|0.3|0.002|0.98|0.2928 ± (0.0159)|0.2681 ± (0.0055)|
|relu|0.3|0.002|0.95|0.1857 ± (0.0229)|0.2220 ± (0.0064)|
|relu|0.3|0.01|0.99|0.3923 ± (0.0211)|0.3062 ± (0.0064)|
|relu|0.3|0.01|0.98|0.3862 ± (0.0327)|0.3092 ± (0.0084)|
|relu|0.3|0.01|0.95|0.3724 ± (0.0359)|0.3092 ± (0.0083)|
|elu|0.2|0.002|0.99|0.3456 ± (0.0274)|0.2967 ± (0.0130)|
|elu|0.2|0.002|0.99|0.2925 ± (0.0175)|0.2711 ± (0.0107)|
|elu|0.2|0.002|0.98|0.1946 ± (0.0259)|0.2238 ± (0.0060)|
|elu|0.2|0.01|0.95|0.3881 ± (0.0224)|0.3101 ± (0.0057)|
|elu|0.2|0.01|0.99|0.3930 ± (0.0286)|0.3149 ± (0.0073)|
|elu|0.2|0.01|0.98|0.3561 ± (0.0172)|0.3057 ± (0.0084)|
|elu|0.3|0.002|0.95|0.3473 ± (0.0199)|0.3014 ± (0.0068)|
|elu|0.3|0.002|0.98|0.2960 ± (0.0150)|0.2697 ± (0.0085)|
|elu|0.3|0.002|0.95|0.1818 ± (0.0211)|0.2200 ± (0.0063)|
|***elu***|***0.3***|***0.01***|***0.99***|***0.4034 ± (0.0160)***|***0.3140 ± (0.0059)***|
|elu|0.3|0.01|0.98|0.3901 ± (0.0262)|0.3116 ± (0.0050)|
|elu|0.3|0.01|0.95|0.3705 ± (0.0266)|0.3073 ± (0.0079)|

#### Results on `musicdata` (batch_size : 50)
We first fixed `batch_size` to 100, and varied the size of embeddings as follows:

|Embedding size|recall@20|ndcg|
|--------------|---------|----|
|100|0.3529 ± (0.0295)|0.2962 ± (0.0103)|
|50|0.3584 ± (0.0181)|0.2876 ± (0.0121)|
|30|0.3327 ± (0.0259)|0.2709 ± (0.010s9)|

The table shows that when the embedding size is 100, the model produces the best accuracy among the results. 
Then, we fixed the embedding size to 100, and checked the effects of activation functions, drop-out, learning rate, and decay ratio as follows:

|Activation|Drop-out|Learning rate|Decay ratio|recall@20|ndcg|
|--------------|---------|----|----|----|----|
|relu|0.2|0.002|0.99|0.3310 ± (0.0229)|0.2772 ± (0.0074)|
|relu|0.2|0.002|0.99|0.2990 ± (0.0257)|0.2597 ± (0.0082)|
|relu|0.2|0.002|0.98|0.2184 ± (0.0181)|0.2147 ± (0.0066)|
|relu|0.2|0.01|0.95|0.3791 ± (0.0246)|0.2969 ± (0.0122)|
|***relu***|***0.2***|***0.01***|***0.99***|***0.3777 ± (0.0164)***|***0.2947 ± (0.0113)***|
|relu|0.2|0.01|0.98|0.3523 ± (0.0177)|0.2868 ± (0.0064)|
|relu|0.3|0.002|0.95|0.3389 ± (0.0239)|0.2808 ± (0.0083)|
|relu|0.3|0.002|0.98|0.2902 ± (0.0258)|0.2565 ± (0.0141)|
|relu|0.3|0.002|0.95|0.2215 ± (0.0134)|0.2130 ± (0.0058)|
|relu|0.3|0.01|0.99|0.3578 ± (0.0183)|0.2956 ± (0.0100)|
|relu|0.3|0.01|0.98|0.3694 ± (0.0116)|0.2947 ± (0.0096)|
|relu|0.3|0.01|0.95|0.3516 ± (0.0263)|0.2863 ± (0.0100)|
|elu|0.2|0.002|0.99|0.3333 ± (0.0371)|0.2712 ± (0.0159)|
|elu|0.2|0.002|0.99|0.2849 ± (0.0247)|0.2506 ± (0.0131)|
|elu|0.2|0.002|0.98|0.2110 ± (0.0189)|0.2132 ± (0.0069)|
|elu|0.2|0.01|0.95|0.3704 ± (0.0181)|0.2914 ± (0.0081)|
|elu|0.2|0.01|0.99|0.3742 ± (0.0273)|0.2926 ± (0.0073)|
|elu|0.2|0.01|0.98|0.3601 ± (0.0289)|0.2877 ± (0.0126)|
|elu|0.3|0.002|0.95|0.3200 ± (0.0292)|0.2725 ± (0.0113)|
|elu|0.3|0.002|0.98|0.2845 ± (0.0243)|0.2482 ± (0.0117)|
|elu|0.3|0.002|0.95|0.2116 ± (0.0232)|0.2111 ± (0.0101)|
|elu|0.3|0.01|0.99|0.3911 ± (0.0265)|0.2886 ± (0.0120)|
|elu|0.3|0.01|0.98|0.3694 ± (0.0236)|0.2915 ± (0.0096)|
|elu|0.3|0.01|0.95|0.3544 ± (0.0279)|0.2801 ± (0.0104)|

#### Results on `moviedata` (batch_size : 500)
Since the dataset is the largest among tested data, we checked the effects of only embedding size and learning rate as follows (see other hyperparameters in [here](https://github.com/jbnu-dslab/DGRec-pytorch/blob/master/hyperparameter/musicdata/param.json)):


|Embedding size|Learning rate|recall@20|ndcg|
|--------------|-------------|---------|----|
|**100**|**0.002**|**0.1594 ± (0.0031)**|**0.1955 ± (0.0015)**|
|100|0.01|0.1574 ± (0.0039)|0.1912 ± (0.0024)|
|50|0.002|0.1509 ± (0.0021)|0.1885 ± (0.0009)|
|50|0.01|0.1586 ± (0.0033)|0.1901 ± (0.0023)|
|50|0.05|0.0921 ± (0.0080)|0.1473 ± (0.0040)|
|30|0.002|0.1349 ± (0.0029)|0.1748 ± (0.0014)|


## References
[1] Song, W., Xiao, Z., Wang, Y., Charlin, L., Zhang, M., & Tang, J. (2019, January). Session-based social recommendation via dynamic graph attention networks. In Proceedings of the Twelfth ACM international conference on web search and data mining (pp. 555-563).
