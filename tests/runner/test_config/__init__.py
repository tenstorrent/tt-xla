from .test_config_inference import test_config as test_config_inference
from .test_config_training import test_config as test_config_training

test_config = test_config_inference | test_config_training