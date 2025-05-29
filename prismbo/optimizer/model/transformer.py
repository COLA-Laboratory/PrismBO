import copy
import numpy as np
from typing import Dict, Hashable, Union, Sequence, Tuple, List

from prismbo.optimizer.model.model_base import Model
from prismbo.agent.registry import model_registry

@model_registry.register("Transformer")
class Transformer(Model):
    def __init__(self):
        super().__init__()
        
    