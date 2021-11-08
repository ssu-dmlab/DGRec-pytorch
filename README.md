
# DGRec-pythorch

Pytorch implementation model of 'Session-Based Social Recommendation via Dynamic Graph Attention Network' which is implemented by Tensorflow.

## Dependencies


## How to run (terminal)
1. `pip install -r requirements.txt`
2. `unzip datasets/bookdata.zip' -d datasets/bookdata/` or `unzip datasets/musicdata.zip' -d datasets/musicdata/`
3. `python -m main`

  
## Arguments
- 활용된 arguments

|Arguments|Explanation|Default|
|------|---|---|
|model|name of model|'DGRec'|
|data_name|name of data|'bookdata'|
|seed|seed number|0|
|epochs|train 반복횟수|20|
|act|activation function|'relu'|
|batch_size|size of batch|100|
|learning_rate|learning rate of model|0.002|
|embedding_size|size of item and user embedding|100|
|max_length|각 user_embedding에 반영 할 수 있는 최대 아이템의 갯수|20|
|samples_1|number of target user's friends|10|
|samples_2|number of target user's friends' friends|5|
|dropout|dropout rate|0.2|


## Data Explanation
### Input data:
* train.tsv: includes user historical behaviors, which is organized by pandas.Dataframe in five fields (SessionId UserId ItemId Timestamps TimeId).
* valid.tsv: the same format as train.tsv, used for tuning hyperparameters.
* test.tsv: the same format as test.tsv, used for testing model.
* adj.tsv: includes links between users, which is also organized by pandas.Dataframe in two fields (FromId, ToId).
* latest_session.tsv: serves as 'reference' to target user. This file records all users available session at each time slot. For example, at time slot t, it stores user u's t-1 th session.
* user_id_map.tsv: maps original string user id to int.
* item_id_map.tsv: maps original string item id to int.

### Douban data:
The statistics of Douban datasets are summarized as follows:

|Dataset|#user|#item|#event|
|------|---|---|---|
|DoubanMusic|39,742|164,223|1,792,501|
|DoubanBook|46,548|212,995|1.908.081|

## Experiments
- 원본 코드와 pytorch로 구현한 코드 비교

* [ ] 원본 / 구현

|data|recall@20|ndcg|
|------|---|---|
|book data|0.3771 / 0.3619|0.2841 / 0.2911|
|music data|0.3382 / 0.3431|0.2539 / 0.2939|

 ## References
 [1] Song, W., Xiao, Z., Wang, Y., Charlin, L., Zhang, M., & Tang, J. (2019, January). Session-based social recommendation via dynamic graph attention networks. In Proceedings of the Twelfth ACM international conference on web search and data mining (pp. 555-563).
