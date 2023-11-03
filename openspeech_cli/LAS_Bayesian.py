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
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args


@hydra.main(config_path=os.path.join("..", "openspeech", "configs"), config_name="train")
def hydra_main(configs: DictConfig) -> None:
    
    # 하이퍼파라미터 탐색 공간 정의
    param_space = {
    'num_encoder_layers': Integer(low=2, high=4, name='num_encoder_layers'),
    'num_decoder_layers': Integer(low=1, high=3, name='num_decoder_layers'),
    'hidden_state_dim': Categorical([128, 256, 512], name='hidden_state_dim'),
    'encoder_dropout_p': Categorical([0.1, 0.2, 0.3, 0.4], name='encoder_dropout_p'),
    'decoder_dropout_p': Categorical([0.1, 0.2, 0.3, 0.4], name='decoder_dropout_p'),
    'optimizer': Categorical(['adam', 'adamw', 'adamp'], name='optimizer'),
    }
    
    # 목적 함수 정의
    @use_named_args(param_space.values())
    def objective(**params):
        # params를 사용하여 configs 수정
        configs.model.num_encoder_layers = int(params['num_encoder_layers'])
        configs.model.num_decoder_layers = int(params['num_decoder_layers'])
        configs.model.hidden_state_dim = int(params['hidden_state_dim'])
        configs.model.encoder_dropout_p = float(params['encoder_dropout_p'])
        configs.model.decoder_dropout_p = float(params['decoder_dropout_p'])
        configs.model.optimizer = str(params['optimizer'])
        # configs를 사용하여 모델, 데이터, 트레이너 설정 및 학습 진행
        
        # 모델, 데이터, 트레이너 설정
        pl.seed_everything(configs.trainer.seed)
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
        model_performance = test_results[0]['test_loss']
        
        # 모델 성능 반환
        return model_performance

    
    # 베이지안 최적화 수행
    # gp_minimize : Gaussian Process 기반 최적화 함수
    # forest_minimize : 랜덤 포레스트(Random Forest) 또는 극단적으로 무작위 트리(Extremely Randomized Trees) 기반 최적화 함수
    # gbrt_minimize : 그레디언트 부스팅 회귀 트리(Gradient Boosted Regression Trees) 기반 최적화 함수
    # dummy_minimize : 단순 무작위 샘플링을 통한 최적화 함수
    res = gp_minimize(objective, dimensions=list(param_space.values()), n_calls=10)
    
    # 최적의 하이퍼파라미터 및 성능 출력 / best_performance의 값이 낮을수록 성능이 좋음
    best_params = "Best Parameters: " + str(res.x) + "\n"
    best_performance = "Best Performance: " + str(res.fun) + "\n"

    # 파일명 생성
    file_number = 1
    filename = 'result.txt'

    # 동일한 이름의 파일이 존재하는지 확인하고, 존재한다면 넘버링 추가
    while os.path.exists(filename):
        file_number += 1
        filename = f'result{file_number}.txt'

    # 결과를 txt 파일에 저장
    with open(f'./result_folder/Bayesian_result/{filename}', 'w') as f:
        f.write(best_params)
        f.write(best_performance)


if __name__ == "__main__":
    hydra_train_init()
    hydra_main()
