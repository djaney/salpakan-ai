from .salpakan import SalpakanEnv
import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Salpakan-v0',
    entry_point='envs:SalpakanEnv',
)