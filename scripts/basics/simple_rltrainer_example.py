import os

from ray.air.config import RunConfig, ScalingConfig
from ray.rllib.algorithms.pg import PGConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.train.rl import RLTrainer

if __name__ == '__main__':
    env_name = 'CartPole-v1'
    algo = "PPO"
    experiment_name = '-'.join((os.path.basename(__file__).split('.py')[0],
                                algo, env_name))
    
    algo = "PPO"
    config_dict = {"PG": PGConfig, "PPO": PPOConfig}
    algo_config = config_dict[algo]()
    algo_config.resources(num_gpus=0)
    algo_config.framework(framework='tf2', eager_tracing=True)
    algo_config.environment(env=env_name,
                            disable_env_checking=True)
    algo_config.rollouts(num_rollout_workers=1)
    algo_config.training(lr=1e-2)
    algo_config.evaluation(evaluation_duration=1,
                           evaluation_num_workers=1,
                           evaluation_duration_unit='episodes',
                           evaluation_interval=5,
                           evaluation_config={"render_env": True,
                                              "explore": False})
    algo_config.debugging(log_level='INFO')
    algo_config.offline_data(output=os.environ['RAY_DATA_DIR'])
    
    trainer = RLTrainer(
        run_config=RunConfig(
            stop={"training_iteration": 100,
                  },
            # name=experiment_name,  # 绝对不能给name，否则tensorboard看不了
            # log_to_file=(xx,xx), # 设置了也没用
        ),
        scaling_config=ScalingConfig(
            num_workers=2,
            use_gpu=False,
        ),
        algorithm=algo,
        config=algo_config.to_dict(),
    )
    result = trainer.fit()
