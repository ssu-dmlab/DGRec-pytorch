
# DGRec-pythorch


## Overview
Pytorch implementation model of 'Session-Based Social Recommendation via Dynamic Graph Attention Network' which is implemented by Tensorflow.


## How to run (terminal)
1. `pip install -r requirements.txt`
2. `unzip datasets/bookdata.zip' -d datasets/bookdata/` or `unzip datasets/musicdata.zip' -d datasets/musicdata/`
3. `python -m main`


## Data
* [ ] Individual interest
- Input_x : user가 Timeid(session)에서 소비한 Itemid -학습 데이터
    * [batch_size, max_length]
- Input_y : user가 Timeid(session)에서 소비한 Itemid -정답 레이블
    * [batch_size, max_length]
- mask_y : input_y에서 소비한 item이 있으면 True, 없으면 False로 나타낸 리스트
    * [batch_size, max_length]
* [ ] Friends' interest (long-term)
- support_nodes_layer1 : friends' friends의 Userid
    * [batch_size * samples_1 * samples_2]
- support_nodes_layer2 : user와 연결되어있는 friends의 Userid
    * [batch_size * samples_2]
* [ ] Friends' interest (short-term)
- support_sessions_layer1 : friends' friends가 가장 최근 Timeid에서 소비한 Itemid
    * [batch_size * samples_1 * samples_2]
- support_sessions_layer2 : user와 연결되어있는 friends가 가장 최근 Timeid에서 소비한 Itemid
    * [batch_size * samples_2]
- support_lengths_layer1 : support_sessions_layer1에서 소비한 item의 갯수
    * [batch_size * samples_1 * samples_2]
- support_lengths_layer2 : support_sessions_layer2에서 소비한 item의 갯수
    * [batch_size * samples_2]
  
## Arguments
- 활용된 arguments

|Arguments|Explanation|Default|
|------|---|---|
|model|모델이름|'DGRec'|
|data_name|데이터이름|'bookdata'|
|seed|seed number|0|
|epochs|train 반복횟수|20|
|act|activation function|'relu'|
|batch_size|size of batch|100|
|learning_rate|learning rate|0.002|
|embedding_size|size of item and user embedding|100|
|max_length|output length|20|
|samples_1|number of friends|10|
|samples_2|number of friends' friends|5|
|dropout|dropout|0.2|


## Experiments
- 원본 코드와 pytorch로 구현한 코드 비교

* [ ] 원본 / 구현

|data|recall@20|ndcg|
|------|---|---|
|book data|0.3771 / 0.3619|0.2841 / 0.2911|
|music data|0.3382 / 0.3431|0.2539 / 0.2939|
