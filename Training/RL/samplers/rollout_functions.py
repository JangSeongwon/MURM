from functools import partial
import numpy as np
import copy

create_rollout_function = partial


def rollout(  
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        preprocess_obs_for_policy_fn=None,
        get_action_kwargs=None,
        return_dict_obs=False,
        full_o_postprocess_func=None,
        reset_callback=None,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        def preprocess_obs_for_policy_fn(x): return x
    raw_obs = []
    raw_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    dones = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0

    '''print('AGENT == GaussianPolicy', agent)'''
    agent.reset()
    # print('Right before env reset from Contextual env in *rollout function*')
    o = env.reset() #Contextual env: def-reset()
    print('Collecting in Rollouts')
    # print('Initial o from contextual env in rollout', o)

    if reset_callback:
        print('NO')
        reset_callback(env, agent, o)
    if render:
        print('NO')
        env.render(**render_kwargs)

    while path_length < max_path_length:
        raw_obs.append(o)
        o_for_agent = preprocess_obs_for_policy_fn(o)

        # print('o for agent', o_for_agent, len(o_for_agent))
        a, agent_info = agent.get_action(o_for_agent, **get_action_kwargs)

        if full_o_postprocess_func:
            full_o_postprocess_func(env, agent, o)

        next_o, r, done, env_info = env.step(copy.deepcopy(a))

        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        terminal = False
        if done:
            # terminal=False if TimeLimit caused termination
            if not env_info.pop('TimeLimit.truncated', False):
                terminal = True
        terminals.append(terminal)
        dones.append(done)
        actions.append(a)
        next_observations.append(next_o)
        raw_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if done:
            break
        o = next_o

        # if path_length == 273:
        #     see = env.checking_final_pos_obj()
        #     print('checking final obj pos',see)
    actions = np.array(actions)

    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)

    observations = np.array(observations)
    next_observations = np.array(next_observations)

    if return_dict_obs:
        observations = raw_obs
        next_observations = raw_next_obs
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        dones=np.array(dones).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        full_observations=raw_obs,
        full_next_observations=raw_obs,
    )

def contextual_rollout(
        env,
        agent,
        observation_keys=None,
        context_keys_for_policy=None,
        obs_processor=None,
        **kwargs
):
    if context_keys_for_policy is None:
        context_keys_for_policy = ['context']
    # print('Going through here')
    paths = rollout(
        env,
        agent,
        preprocess_obs_for_policy_fn=obs_processor,
        **kwargs
    )

    return paths

