import os

from ray.air.config import RunConfig, ScalingConfig
from ray.train.rl import RLTrainer

if __name__ == '__main__':
    env_name = 'CartPole-v0'
    algorithm = "PG"
    experiment_name = '-'.join((os.path.basename(__file__).split('.py')[0],
                                algorithm, env_name))
    
    trainer = RLTrainer(
        run_config=RunConfig(
            stop={"training_iteration": 5},
            # name=experiment_name,  # 绝对不能给name，否则tensorboard看不了
            # log_to_file=(xx,xx), # 设置了也没用
        ),
        scaling_config=ScalingConfig(
            num_workers=2,
            use_gpu=False,
        ),
        algorithm=algorithm,
        config={
            "output": os.environ['RAY_DATA_DIR'],
            "env": env_name,
            "log_level": "INFO",
            "framework": "tf2",
            "evaluation_num_workers": 1,
            "evaluation_interval": 5,
            "evaluation_duration": 1,
            "evaluation_duration_unit": "episodes",
            "evaluation_config": {
                "input": "sampler",
                "render_env": True,
                "explore": False,
            },
        },
    )
    result = trainer.fit()
