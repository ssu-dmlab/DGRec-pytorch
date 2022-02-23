import os
import fire
import sys
import json
from dotmap import DotMap
from src.main_trainer import main

os.chdir('./src')


def main_wrapper(data_name='bookdata'):
    param_path = f'../hyperparameter/{data_name}/param.json'

    with open(param_path, 'r') as in_file:
        param = DotMap(json.load(in_file))

    main(model=param.model,
         data_name=param.data_name,
         seed=param.seed,
         epochs=param.epochs,
         act=param.act,
         batch_size=param.batch_size,
         learning_rate=param.learning_rate,
         embedding_size=param.embedding_size,
         max_length=param.max_length,
         samples_1=param.samples_1,
         samples_2=param.samples_2,
         dropout=param.dropout,
         decay_rate=param.decay_rate,
         )


if __name__ == "__main__":
    sys.exit(fire.Fire(main_wrapper))
