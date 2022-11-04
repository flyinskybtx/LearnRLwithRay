import argparse
import os

import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.pg import PG, PGConfig

from logs import LOG_DIR

config_dict = {"PG": PGConfig, "PPO": PPOConfig}


def train_fn(config, reporter):
    iterations = config.get('train-iterations', 10)
    algo_config = config_dict[algo]()
    # algo_config.resources(num_gpus=0)
    algo_config.framework(framework='tf2', eager_tracing=True)
    algo_config.environment(env=config.get('env_name'),
                            disable_env_checking=True)
    algo_config.rollouts(num_rollout_workers=config.get("num_workers", 1))
    algo_config.training(lr=config.get("lr", 1e-2))
    algo_config.evaluation(evaluation_duration=1,
                           evaluation_duration_unit='episodes',
                           evaluation_interval=5,
                           evaluation_config={"render_env": True,
                                              "explore": False})
    
    trainer = algo_config.build()
    for _ in range(iterations):
        result = trainer.train()
        # result["phase"] = 1
    #     reporter(**result)
    #     phase1_time = result["timesteps_total"]
    # print("phase1 time:", phase1_time)
    state = trainer.save()
    trainer.stop()


if __name__ == '__main__':
    env_name = 'CartPole-v1'
    algo = "PPO"
    experiment_name = '-'.join((os.path.basename(__file__).split('.py')[0],
                                algo, env_name))
    config = {
        "train-iterations": int(1e4),
        "algo": algo,
        "env_name": env_name,
        "num_gpus": 0,
        "num_workers": 2,
        "lr": 1e-2,
    }
    checkpoint_config = air.CheckpointConfig(
        num_to_keep=5,
        checkpoint_frequency=5,
        checkpoint_score_attribute='episode_reward_mean',
        checkpoint_at_end=True
    )
    
    os.makedirs(os.path.join(LOG_DIR, experiment_name), exist_ok=True)
    run_config = air.RunConfig(
        stop={
            "training_iteration": config["train-iterations"] // 10,
            "episode_reward_mean": 200,
        },
        name=experiment_name,
        local_dir=os.environ["TUNE_RESULT_DIR"],
        checkpoint_config=checkpoint_config,
        verbose=2,
    )
    # --------------
    ray.init()
    resources = eval(algo).default_resource_request(config)
    tuner = tune.Tuner(
        # trainable=train_fn,
        tune.with_resources(
            trainable=train_fn,
            resources=tune.PlacementGroupFactory(
                [{'CPU': 1.0}] + [{'CPU': 1.0}] * config["num_workers"]
            )
        ),
        param_space=config,
        run_config=run_config,
    )
    tuner.fit()
