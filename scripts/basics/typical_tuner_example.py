import os

from ray import air, tune

from logs import LOG_DIR

if __name__ == '__main__':
    env_name = 'CartPole-v0'
    algorithm = "PG"
    experiment_name = '-'.join((os.path.basename(__file__).split('.py')[0],
                                algorithm, env_name))
    
    checkpoint_config = air.CheckpointConfig(
        num_to_keep=5,
        checkpoint_frequency=5,
        checkpoint_score_attribute='episode_reward_mean',
        checkpoint_at_end=True
    )
    
    log_to_file = (os.path.join(LOG_DIR, experiment_name, 'stdout'),
                   os.path.join(LOG_DIR, experiment_name, 'stderr'))
    os.makedirs(os.path.join(LOG_DIR, experiment_name), exist_ok=True)
    run_config = air.RunConfig(
        stop={
            "training_iteration": int(1e4),
            "episode_reward_mean": 300.0
        },
        name=experiment_name,
        local_dir=os.environ["TUNE_RESULT_DIR"],
        log_to_file=log_to_file,
        # callbacks=[],
        checkpoint_config=checkpoint_config,
        verbose=2,
    )
    
    config = {
        "output": os.environ['RAY_DATA_DIR'],
        "num_workers": 2,
        "num_gpus": 0,
        "evaluation_config": {
            # "input": "sampler",
            "render_env": True,
            "explore": False,
        },
        "evaluation_num_workers": 1,
        "evaluation_interval": 5,
        "evaluation_duration": 1,
        "evaluation_duration_unit": "episodes",
        "env": env_name,
        "framework": "tf2",
        "eager_tracing": True,
        "log_level": "INFO",
        "disable_env_checking": True,
    }
    
    ray_results = os.environ["TUNE_RESULT_DIR"]
    if experiment_name in os.listdir(ray_results):
        checkpoint_path = os.path.join(ray_results, experiment_name)
        tuner = tune.Tuner.restore(checkpoint_path)
        print(" --- Restore from checkpoint. --- ")
    
    else:
        tuner = tune.Tuner(
            trainable=algorithm,  # 不能用rltrainer
            run_config=run_config,
            param_space=config,
        )
        print(" --- Train from beginning. --- ")
    
    result = tuner.fit()
