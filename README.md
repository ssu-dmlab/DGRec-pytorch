
# DGRec-pythorch

## Overview
Tensorflow로 구현된 'Session-Based Social Recommendation via Dynamic Graph Attention Network'의 source code를 Pytorch로 구현하였다.

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
    
## Project structure
`DGRec-pytorch`는 다음과 같은 구조로 이루어져있다. `DGRec-pytorch` 코드의 세부 내용을 분석할 때는 `main.py`를 시작점으로 해서 살펴보면 된다. 

```shell
├── README.md
├── datasets                        # datasets을 저장하는 폴더
│   └── Douban                       
└── src                             # source codes를 저장하는 폴더
    ├── data.py                     # datasets (DataSet, DataLoader) 관련 작업을 처리하는 script
    ├── experiments                 # 실험 scripts를 저장하는 폴더
    │    ├── __init__.py
    │    └── exp_hyper_param.py
    ├── main.py                     # 사용자 입력을 처리하는 script
    ├── models                      # model의 코드를 저장하는 폴더 (여러 모델을 구현한다고 가정)
    │    ├── __init__.py
    │    └── DGRec                # 'DGRec'의 코드를 저장하는 폴더
    │        ├── __init__.py
    │        ├── model.py           # DGRec 구현 담당 (주로 forward 함수 구현)
    │        ├── train.py           # data, hyper_param을 받아 DGRec 훈련 담당 (주로 gradient descent & backprop)
    │        └── eval.py            # test data에 대해 훈련된 DGRec 평가 담당 (주로 정확도 측정)
    └── utils.py
```
