
from gym.envs.registration import register


register(
    id='IntersectionEnv-v2',
    entry_point='envs.IntersectionEnv:IntersectionEnv',
)
