import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint   # deepspeed: Microsoft가 만든 대규모 모델 학습 최적화 라이브러리
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from transformers import AutoProcessor, AutoTokenizer

from simlingo_training.config import TrainConfig
from simlingo_training.utils.logging_project import setup_logging
# from simlingo_training.callbacks.visualise import VisualiseCallback

@hydra.main(config_path=f"config", config_name="config", version_base="1.1")
def main(cfg: TrainConfig):
    
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(42)
    
    # eval_mode = "QA"
    # eval_mode = "commentary"
    eval_mode = "Driving"  # LOCAL EDIT

    qa_dataset = cfg.data_module.qa_dataset
    insteval_dataset = cfg.data_module.insteval_dataset
    load_path = '/home/ailab/git/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt'  # LOCAL EDIT
    if load_path is not None:
        load_path_config = Path(load_path).parent.parent / '.hydra/config.yaml'
        cfg = OmegaConf.load(load_path_config)
        cfg.data_module.base_dataset.data_path = 'database/simlingo'  # LOCAL EDIT: NAS layout
        cfg.data_module.base_dataset.bucket_path = 'database/simlingo'  # LOCAL EDIT
        cfg.checkpoint = load_path  # LOCAL EDIT: raw state dict, not a Lightning ckpt
    
    cfg.data_module.qa_dataset = qa_dataset
    cfg.data_module.insteval_dataset = insteval_dataset
    cfg.gpus = 1
    cfg.data_module.num_workers = 0  # LOCAL EDIT: debug
    cfg.data_module.batch_size = 1  # LOCAL EDIT: debug

    print(f'Eval mode: {eval_mode}')
    print(f'Checkpoint: {load_path}')
    print(f"Using {cfg.gpus} GPUs")
    
    if eval_mode == "QA" or eval_mode == "commentary":
        cfg.data_module.dreamer_dataset = None
        cfg.data_module.driving_dataset = None
        cfg.data_module.insteval_dataset = None 
    elif eval_mode == "Dreaming":
        cfg.data_module.dreamer_dataset = None
        cfg.data_module.driving_dataset = None
        cfg.data_module.qa_dataset = None
    elif eval_mode == "Driving":  # LOCAL EDIT
        cfg.data_module.dreamer_dataset = None
        cfg.data_module.qa_dataset = None
        cfg.data_module.insteval_dataset = None
    
    if eval_mode == "QA":
        cfg.data_module.base_dataset.use_commentary = False
        cfg.data_module.base_dataset.use_qa = True
    elif eval_mode == "commentary":
        cfg.data_module.base_dataset.use_commentary = True
        cfg.data_module.base_dataset.use_qa = False
    elif eval_mode == "Dreaming":
        cfg.data_module.base_dataset.use_safety_flag = True
    elif eval_mode == "Driving":  # LOCAL EDIT
        cfg.data_module.base_dataset.use_safety_flag = False
    
    # disable image augmentation
    cfg.data_module.base_dataset.img_augmentation = False
    
    # disable img_shift_augmentation
    cfg.data_module.base_dataset.img_shift_augmentation = False
    
    if "2B" in cfg.model.language_model.variant:
        processor = AutoTokenizer.from_pretrained(cfg.model.language_model.variant, trust_remote_code=True, use_fast=False)
    else:
        processor = AutoProcessor.from_pretrained(cfg.model.language_model.variant, trust_remote_code=True, use_fast=False)     # 모델 입력과 출력 주변의 형식을 맞춰주는 전처리/후처리 객체
    model_type_name = cfg.model.vision_model.variant.split('/')[1]
    cache_dir = f"pretrained/{(model_type_name)}"
    
    data_module = hydra.utils.instantiate(  # Hydra 설정을 사용해서 추론용 DataModule 생성; tokenizer와 모델 규격 정보를 전달
        cfg.data_module, 
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant, # 이미지 입력 처리에 사용할 vision encoder 이름
        llm_variant=cfg.model.language_model.variant,   # 텍스트/언어 처리에 사용할 language model 이름
        predict=True,                                   # 추론 모드로 설정
        _recursive_=False                               # cfg.data_module 내부의 하위 설정들은 재귀적으로 instantiate하지 않음; 즉, cfg.data_module 내부의 하위 설정들은 그대로 전달
    )
    
    model = hydra.utils.instantiate(    # Hydra 설정으로 실제 DrivingModel 객체를 생성하는 코드
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=cache_dir,    # 다운로드한 pretrained 모델 파일을 재사용하기 위한 로컬 저장 공간
        _recursive_=False       # cfg.model 내부의 하위 설정들은 재귀적으로 instantiate하지 않음; 즉, cfg.model 내부의 하위 설정들은 그대로 전달
        )

    if cfg.checkpoint is not None:
        if os.path.isdir(cfg.checkpoint):
            state_dict = get_fp32_state_dict_from_zero_checkpoint(cfg.checkpoint)
        else:
            state_dict = torch.load(cfg.checkpoint, map_location="cpu")     # state_dict: PyTorch가 모델의 가중치를 저장하는 dictionary
        model.load_state_dict(state_dict)   # 모델에 checkpoint의 가중치(Tensor)를 복사

        
    # print config
    print(OmegaConf.to_yaml(cfg))
    os.environ["WANDB_DISABLE_CODE"] = "True"

    
    # setup logging
    setup_logging(cfg)

    # resume training
    resume_path = "./checkpoints/last.ckpt"


    if os.path.exists(resume_path) and cfg.resume:
        resume_path = resume_path
    else:
        resume_path = None
    
    # setup lightning logger
    loggers = []

    strategy = cfg.strategy
    if strategy == "deepspeed_stage_2":
        strategy = pl.strategies.DeepSpeedStrategy(
            stage=2, loss_scale=cfg.fp16_loss_scale, logging_batch_size_per_gpu=cfg.data_module.batch_size
        )
  
    print(f"Number of GPUS: {cfg.gpus}")
    overfit = 0
    
    if cfg.gpus >= 1:
        trainer = Trainer(
            accelerator="gpu",
            benchmark=True,
            devices=cfg.gpus,           # 사용할 gpu 개수
            gradient_clip_val=0.3,
            log_every_n_steps=20,
            logger=loggers,
            precision=cfg.precision,    # 계산 정밀도
            strategy=strategy,
            sync_batchnorm=True,
            max_epochs=cfg.max_epochs,
            overfit_batches=overfit,
            check_val_every_n_epoch=cfg.val_every_n_epochs,
        )

    trainer.predict(model, data_module)  # predict: 전체 prediction 작업을 시작하는 Trainer의 메서드

if __name__ == "__main__":
    main()