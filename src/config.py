from abc import ABC
import dataclasses
from ruamel.yaml import YAML


@dataclasses.dataclass
class BaseConfig(ABC):
    def save(self, file_path):
        yaml = YAML()
        with open(file_path, 'w') as f:
            yaml.dump(dataclasses.asdict(self), f)

    @classmethod
    def load(cls, file_path, **kwargs):
        yaml = YAML()
        with open(file_path) as f:
            config_dict = yaml.load(f)
        config_dict.update(kwargs)

        fields = tuple(map(lambda field: field.name, dataclasses.fields(cls)))
        config_dict = {k: v for k, v in config_dict.items() if k in fields}

        return cls(**config_dict)

    def __post_init__(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            value = field.type(value)
            setattr(self, field.name, value)


@dataclasses.dataclass
class Config(BaseConfig):
    #task
    discount: float = .99
    disclam: float = .95
    horizon: int = 15
    kl_scale: float = 1.
    expl_scale: float = .3
    alpha: float = .8
    rho: float = 0.
    eta: float = 1e-4
    action_repeat: int = 2

    #model
    actor_layers: tuple = (256, 256)
    critic_layers: tuple = (256, 256)
    wm_layers: tuple = (200, 200)
    deter_dim: int = 164
    stoch_dim: int = 24
    obs_emb_dim: int = 32

    pn_layers: tuple = (64, 128)
    pn_depth: int = 32
    pn_number: int = 600
    pn_dropout: float = 0.

    #train
    actor_lr: float = 8e-5
    critic_lr: float = 8e-5
    critic_polyak: float = .99
    wm_lr: float = 6e-4
    batch_size: int = 50
    max_grad: float = 100.
    seq_len: int = 50
    buffer_capacity: int = 5*10**2
    training_steps: int = 400
    eval_freq: int = 10000

    device: str = 'cuda'
    observe: str = 'states'
    task: str = 'walker_stand'
    logdir: str = 'logdir/tmp'
