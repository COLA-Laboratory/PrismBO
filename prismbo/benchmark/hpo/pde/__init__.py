from .diffusion import Diffusion2D
from .convection import Convection2D
from .reaction import Reaction2D
from .wave import Wave2D
from .burgers import Burgers2D

PROBLEMS = [
    'Burgers2D',
    'Diffusion2D',
    'Convection2D',
    'Reaction2D',
    'Wave2D',
]

pde_classes = {
    'Burgers2D': Burgers2D,
    'Diffusion2D': Diffusion2D,
    'Convection2D': Convection2D,
    'Reaction2D': Reaction2D,
    'Wave2D': Wave2D,
}