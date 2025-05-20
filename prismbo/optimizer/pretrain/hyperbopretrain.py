from prismbo.agent.registry import pretrain_registry
from prismbo.optimizer.pretrain.pretrain_base import PretrainBase



@pretrain_registry.register("hyperbo")
class HyperBOPretrain(PretrainBase):
    def __init__(self, config) -> None:
        super().__init__(config)