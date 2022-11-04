import os

import ray
from ray.rllib.utils import try_import_tf

_, tf, version = try_import_tf()

import ray.rllib.agents.ppo as ppo
from ray import air, tune
from ray.rllib.algorithms import pg
from ray.tune import ExperimentAnalysis

from models import MODEL_DIR


def make_reload_trainer(alg, model_weight_h5):
    class PPOReload(ppo.PPOTrainer):
        def __init__(self, config, **kwargs):
            super(PPOReload, self).__init__(config, **kwargs)
            """ Needs full path here!"""
            self.get_policy().model.base_model.load_weights(model_weight_h5)
            self.workers.sync_weights()  # Important!!!
        
        def reset_config(self, new_config):
            """ to enable reuse of actors """
            self.config = new_config
            return True
    
    class PGReload(pg.PG):
        def __init__(self, config, **kwargs):
            super(PGReload, self).__init__(config, **kwargs)
            """ Needs full path here!"""
            self.get_policy().model.base_model.load_weights(model_weight_h5)
            self.workers.sync_weights()  # Important!!!
        
        def reset_config(self, new_config):
            """ to enable reuse of actors """
            self.config = new_config
            return True
    
    reload_dict = {'PPO': PPOReload, 'PG': PGReload, }
    return reload_dict[alg]


alg_dict = {'PG': (pg.PG, pg.PGConfig().to_dict()),
            'PPO': (ppo.PPOTrainer, ppo.PPOConfig().to_dict()),
            }

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    ray.init(num_cpus=4, num_gpus=0, local_mode=True)
    # for gpu in gpus:
    #     tf.config.set_logical_device_configuration(
    #         gpu, [tf.config.experimental.VirtualDeviceConfiguration(
    #             memory_limit=1024)])  # 限制GPU用量
    # tf.config.experimental.set_memory_growth(gpu, True)
    
    env_name = 'CartPole-v1'
    algorithm = "PPO"
    experiment_name = '-'.join((os.path.basename(__file__).split('.py')[0],
                                algorithm, env_name))
    ray_results_dir = os.environ["TUNE_RESULT_DIR"]
    experiment_config = {
        "num_workers": 0,
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
        "eager_tracing": False,
        "log_level": "INFO",
        "disable_env_checking": False,
    }
    checkpoint_config = air.CheckpointConfig(
        num_to_keep=5, checkpoint_frequency=5,
        checkpoint_score_attribute='episode_reward_mean',
        checkpoint_at_end=True
    )
    trainer, config = alg_dict[algorithm]
    config.update(experiment_config)
    print(config)
    
    # --- Phase_1 训练第一次
    phase_1_name = experiment_name + '_phase_1'
    if phase_1_name in os.listdir(ray_results_dir):
        checkpoint_path = os.path.join(ray_results_dir, phase_1_name)
        tuner = tune.Tuner.restore(checkpoint_path)
        print(" --- Restore from checkpoint. --- ")
    
    else:
        tuner = tune.Tuner(
            # tune.with_resources(trainer, resources={"CPU": 1, "GPU": 0.5}),
            trainer,
            run_config=air.RunConfig(
                stop={"training_iteration": 50,
                      "episode_reward_mean": 100},
                name=phase_1_name,
                local_dir=os.environ["TUNE_RESULT_DIR"],
                checkpoint_config=checkpoint_config,
                verbose=3,
            ),
            param_space=config,
        )
        print(" --- Train from beginning. --- ")
    
    tuner.fit()
    del tuner
    
    # --- Phase_2 保存checkpoint 和 weights
    analysis = ExperimentAnalysis(os.path.join(ray_results_dir, phase_1_name))
    best_trial = analysis.get_best_trial(metric="episode_reward_mean",
                                         mode="max")
    best_ckp = analysis.get_best_checkpoint(best_trial,
                                            metric="episode_reward_mean",
                                            mode="max")
    phase_2_name = experiment_name + '_phase_2'
    agent = trainer(config=config, env="CartPole-v0", )
    agent.restore(best_ckp)
    policy = agent.get_policy()
    model_h5 = os.path.join(MODEL_DIR, experiment_name + '.h5')
    policy.model.base_model.save_weights(model_h5)
    print(" ---\n Saved weights to ", model_h5)
    del agent
    
    # --- Phase_3 读取weights，新开一个训练
    phase_3_name = experiment_name + '_phase_3'
    new_trainer = make_reload_trainer(algorithm, model_h5)
    new_tuner = tune.Tuner(
        tune.with_resources(
            trainable=new_trainer,
            resources=tune.PlacementGroupFactory(
                [{'CPU': 1.0}] + [{'CPU': 1.0}] * config["num_workers"]
            )
        ),
        run_config=air.RunConfig(
            stop={"training_iteration": 30,
                  "episode_reward_mean": 200},
            name=phase_3_name,
            local_dir=os.environ["TUNE_RESULT_DIR"],
            checkpoint_config=checkpoint_config,
        ),
        param_space=config
    )
    print(" --- Phase 3 Train from beginning. --- ")
    new_tuner.fit()
