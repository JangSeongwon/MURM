import gym

BULLET_ENVIRONMENT_SPECS = (
    {
        'id': 'MURMENV-v1',
        'entry_point': ('roboverse.envs.murm_env:MURMENV'),
    },

{
        'id': 'MURMENV-v2',
        'entry_point': ('roboverse.envs.murm_env_v2:MURMENV_v2'),
    },

{
        'id': 'MURMENV-v3',
        'entry_point': ('roboverse.envs.murm_env_v4:MURMENV_v3'),
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
