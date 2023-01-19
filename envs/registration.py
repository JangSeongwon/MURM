import gym

BULLET_ENVIRONMENT_SPECS = (
    {
        'id': 'MURMENV-v0',
        'entry_point': ('roboverse.envs.murm_env:MURMENV'),
    },
    {
        'id': 'MURMENV-v1',
        'entry_point': ('roboverse.envs.murm_env_m1:MURMENV_m1'),
    },
    {
        'id': 'MURMENV-v2',
        'entry_point': ('roboverse.envs.murm_env_m2:MURMENV_m2'),
    },
)

def register_bullet_environments():
    for bullet_environment in BULLET_ENVIRONMENT_SPECS:
        gym.register(**bullet_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in BULLET_ENVIRONMENT_SPECS)
    return gym_ids

def make(env_name, *args, **kwargs):
    env = gym.make(env_name, *args, **kwargs)
    return env
