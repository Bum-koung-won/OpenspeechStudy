# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import hydra
import sentencepiece
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_info
import random
from openspeech.dataclass.initialize import hydra_train_init
from openspeech.datasets import DATA_MODULE_REGISTRY
from openspeech.models import MODEL_REGISTRY
from openspeech.tokenizers import TOKENIZER_REGISTRY
from openspeech.utils import get_pl_trainer, parse_configs
from skopt import gp_minimize, forest_minimize, gbrt_minimize, dummy_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.callbacks import CheckpointSaver

used_seeds = []
used_parameters = set()
train_cycle = 1
best_result = []

def generate_unique_seed():
    while True:
        new_seed = random.randint(0, 1024)  # 새로운 seed 값 생성
        if new_seed not in used_seeds:
            used_seeds.append(new_seed)
            return new_seed


@hydra.main(config_path=os.path.join("..", "openspeech", "configs"), config_name="train")
def hydra_main(configs: DictConfig) -> None:
    param_space = {
    'num_encoder_layers': Integer(low=2, high=4, name='num_encoder_layers'),
    'num_decoder_layers': Integer(low=1, high=3, name='num_decoder_layers'),
    'hidden_state_dim': Categorical([128, 256, 512], name='hidden_state_dim'),
    'num_attention_heads': Categorical([2, 4, 8], name='num_attention_heads'), 
    'encoder_dropout_p': Categorical([0.1, 0.2, 0.3, 0.4], name='encoder_dropout_p'),
    'decoder_dropout_p': Categorical([0.1, 0.2, 0.3, 0.4], name='decoder_dropout_p'),
    'optimizer': Categorical(['adam', 'adamw', 'adamp'], name='optimizer'), 
    # 여기에 다른 하이퍼파라미터들도 추가.
    }
    
    # 목적 함수 정의
    @use_named_args(param_space.values())
    def objective(**params):
        global train_cycle
        global best_result
        current_params = tuple(params.items())
        # 이전에 사용된 파라미터 조합인지 확인
        if current_params in used_parameters:
            return float('inf')  # 중복된 경우, 최악의 성능을 반환하여 해당 조합이 선택되지 않도록 한다.
        else:
            used_parameters.add(current_params)  # 새로운 조합이면, 사용된 파라미터 조합 세트에 추가한다.
        
        
        # params를 사용하여 configs 수정
        configs.model.num_encoder_layers = int(params['num_encoder_layers'])
        configs.model.num_decoder_layers = int(params['num_decoder_layers'])
        configs.model.hidden_state_dim = int(params['hidden_state_dim'])
        configs.model.num_attention_heads = int(params['num_attention_heads'])
        configs.model.encoder_dropout_p = float(params['encoder_dropout_p'])
        configs.model.decoder_dropout_p = float(params['decoder_dropout_p'])
        configs.model.optimizer = str(params['optimizer'])
        
        # 모델, 데이터, 트레이너 설정
        random_seed = generate_unique_seed()  # 중복되지 않는 seed 생성
        pl.seed_everything(random_seed)  # PyTorch Lightning에 seed 설정
        logger, num_devices = parse_configs(configs)
        data_module = DATA_MODULE_REGISTRY[configs.dataset.dataset](configs)
        data_module.prepare_data()
        tokenizer = TOKENIZER_REGISTRY[configs.tokenizer.unit](configs)
        data_module.setup()
        model = MODEL_REGISTRY[configs.model.model_name](configs=configs, tokenizer=tokenizer)
        trainer = get_pl_trainer(configs, num_devices, logger)
        
        # 학습 및 평가
        trainer.fit(model, data_module)
        test_results = trainer.test(model, data_module)
        model_performance = test_results[0]['test_cer']
        model_loss = test_results[0]['test_loss']
        
        with open(result_file_path, 'a') as f:
            f.write(f"{train_cycle}회차 - Loss : {model_loss} 성능: {model_performance}, 시드: {random_seed}, 파라미터: {current_params}\n")  # 회차별 성능, random seed, parameter 값 저장
        
        cycle_performance = [train_cycle, model_loss, model_performance, random_seed, current_params]
        if len(best_result) == 0:
            best_result.append(cycle_performance)
        elif model_performance < best_result[0][2]:
            best_result = []
            best_result.append(cycle_performance)
        
        train_cycle += 1
        # 모델 성능 반환
        return model_performance

    
    result_file_path = f'./result_folder/Bayesian_result/results.txt'
    open(result_file_path, 'w').close()
    
    # 베이지안 최적화 수행
    # gp_minimize : Gaussian Process 기반 최적화 함수
    # forest_minimize : 랜덤 포레스트(Random Forest) 또는 극단적으로 무작위 트리(Extremely Randomized Trees) 기반 최적화 함수
    # gbrt_minimize : 그레디언트 부스팅 회귀 트리(Gradient Boosted Regression Trees) 기반 최적화 함수
    # dummy_minimize : 단순 무작위 샘플링을 통한 최적화 함수
    res = dummy_minimize(objective, dimensions=list(param_space.values()), n_calls=10)
    with open(result_file_path, 'a') as f:
        f.write('\n')
        f.write(f'Best Cycle : {best_result[0][0]}회차\n')
        f.write(f'Best Loss : {best_result[0][1]}\n')
        f.write(f'Best Performance : {best_result[0][2]}\n')
        f.write(f'Best Seed : {best_result[0][3]}\n')
        f.write(f'Best Parameter : {best_result[0][4]}')
    
if __name__ == "__main__":
    hydra_train_init()
    hydra_main()