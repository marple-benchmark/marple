from typing import Optional, List, Dict, Any

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

@dataclass
class ModelArguments:
    """
    Model args.
    """
    # model name 
    model_name: str = "transformer_model"
    # model checkpoint save dir
    n_checkpoints: int = 1 # how often we save checkpoints, curr n_epochs // n_checkpoints
    dirpath: str = "/vision/u/emilyjin/marple_long/checkpoints"
    # args for TransformerHeadPolicyModel 
    transformer_input_dim: int = 256 # was 32
    transformer_hidden_dim: int = 256
    transformer_position_embedding_dim: int = 64 # was 4
    transformer_depth: int = 8
    transformer_n_heads: int = 8
    transformer_dropout: float = 0.1 # was 0.1
    transformer_activation: str = 'gelu'
    # SimpleViT args
    vit_image_size: tuple = (20, 20)
    vit_patch_size: int = 1
    vit_num_classes: int = 1000
    vit_channels: int = 8
    # can change these vit args
    vit_dim: int = 768
    vit_depth: int = 12
    vit_heads: int = 16
    vit_mlp_dim: int = 2048
    # audio transformer args
    max_audio_length: int = 200
    checkpoint_name: str = "checkpoint.ckpt"
    # args for low policy model
    embedding_dim: int = 1024
    subgoal_channel: int = 8
    object_loss: bool = False

@dataclass
class DataArguments:
    """
    Data args
    """
    data_path: Optional[str] = field(
        default="/vision/u/emilyjin/marple_long/data",
        metadata={
            "help": (
                "Path to data."
            )
        },
    )
    train_val_split: Optional[float] = field(
        default=0.7,
        metadata={
            "help": (
                "Train val split."
            )
        },
    )
    split: Optional[str] = field(
        default='train',
        metadata={
            "help": (
                "The split to load: train, test."
            )
        },
    )
    data_level: Optional[str] = field(
        default="mid",
        metadata={
            "help": (
                "Data level: low, mid."
            )
        }
    )
    mission_dict: Optional[Dict[str, float]] = field(
        default_factory=lambda: {
        },
        metadata={
            "help": "List of missions that can be performed by an agent with number of"
            "trajectories for the mission (allows to maniuplate high-level planning)."
        }
    )
    end_state_flag: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to include end state flag in data."
            )
        },
    )
    with_audio: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to include audio in data."
            )
        },
    )
    historical: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to include historical data in data."
            )
        },
    )
    max_audio_length: Optional[int] = field(
        default=200,
        metadata={
            "help": (
                "Max audio length."
            )
        },
    )
    max_atomic_length: Optional[int] = field(
        default=500,
        metadata={
            "help": (
                "Max atomic length."
            )
        },
    )
    goal_conditioned: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to include goal conditioned data."
            )
        },
    )
    room: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Whether to include goal conditioned data."
            )
        },
    )


@dataclass
class DataLoaderArguments:
    batch_size: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "Batch size. Equal to batch_size * gpus * num_nodes."
            )
        },
    )
    num_workers: Optional[int] = field(
        default=8,
        metadata={
            "help": (
                "Num CPU workers for dataloader."
            )
        },
    )
    pin_memory: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Pin memory."
            )
        },
    )

@dataclass
class TrainerArguments:
    """
    Trainer args for pl trainer
    """
    max_epochs: Optional[int] = field(
        default=20,
        metadata={
            "help": (
                "Max epochs."
            )
        },
    )
    gradient_clip_val: Optional[float] = field(
        default=1.0,
        metadata={
            "help": (
                "Gradient clip val."
            )
        },
    )
    num_nodes: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "N nodes (for distributed training). (not N GPUS?)"
            )
        },
    )
    devices: Optional[Any] = field(
        default=1,
        metadata={
            "help": (
                "n gpus"
            )
        },
    )
    deterministic: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Deterministic"
            )
        },
    )
    accelerator: Optional[str] = field(
        default="auto",
        metadata={
            "help": (
                "cpu, gpu, tpu"
            )
        },
    )


@dataclass
class OptimizerArguments:
    """
    Trainer args for optimizer
    """
    name: Optional[str] = field(
        default="adam",
        metadata={
            "help": (
                "Optimizer name."
            )
        },
    )
    lr: Optional[float] = field(
        default=1e-4,
        metadata={
            "help": (
                "Learning rate."
            )
        },
    )
    weight_decay: Optional[float] = field(
        default=0, 
        metadata={
            "help": (
                "Regularizer."
            )
        },
    )
    use_lr_scheduler: Optional[bool] = field(
        default=True, 
        metadata={
            "help": (
                "Regularizer."
            )
        },
    ) 
    lr_restart: Optional[int] = field(
        default=2000, 
        metadata={
            "help": (
                "Regularizer."
            )
        },
    )     
    warmup: Optional[int] = field(
        default=1, 
        metadata={
            "help": (
                "Regularizer."
            )
        },
    )         

@dataclass
class WandBArguments:
    """
    Wandb args
    """
    project: Optional[str] = field(
        default="marple-long",
        metadata={
            "help": (
                "Wandb project."
            )
        },
    )
    name: Optional[str] = field(
        default="experiment-0",
        metadata={
            "help": (
                "Name of the run."
            )
        },
    )
    save_dir: Optional[str] = field(
        default="/vision/u/emilyjin/marple_long/wandb/",
        metadata={
            "help": (
                "Where to save cached files."
            )
        },
    )
   
@dataclass
class ExperimentArguments:
    experiment_name: Optional[str] = field(
        default='experiment_0',
        metadata={
            "help": (
                "The experiment we are running, see google docs/paper for details."
            )
        },
    )
    random_seed: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Random seed."
            )
        },
    )
    agent: Optional[str] = field(
        default="A",
        metadata={
            "help": (
                "What agent we are training."
            )
        },
    )
    mission_name: Optional[str] = field(
        default="get_night_snack",
        metadata={
            "help": (
                "What mission the agent is performing."
            )
        },
    )
    mission_1: Optional[str] = field(
        default="feed_dog",
        metadata={
            "help": (
                "What mission agent 1 is performing."
            )
        },
    )    
    mission_2: Optional[str] = field(
        default="do_laundry",
        metadata={
            "help": (
                "What mission agent 2 is performing."
            )
        },
    )       
    mission_1: Optional[str] = field(
        default="feed_dog",
        metadata={
            "help": (
                "What mission agent 1 is performing."
            )
        },
    )    
    mission_1_pref: Optional[float] = field(
        default=1.0,
        metadata={
            "help": (
                "Agent 1 mission pref for their primary mission"
            )
        },
    )        
    mission_2_pref: Optional[float] = field(
        default=1.0,
        metadata={
            "help": (
                "Agent 2 mission pref for their primary mission"
            )
        },
    )   
    results_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Directory to save results"
            )
        },
    )                    

@dataclass
class SimulatorArguments:
    env: Optional[str] = field(
        default="MiniGrid-AutoGenerate-16x16-N2-v1",
        metadata={
            "help": (
                "Simulator environment."
            )
        },
    )
    tile_size: Optional[int] = field(
        default=32,
        metadata={
            "help": (
                "Tile size."
            )
        },
    )
    save: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "Save gif/wav/text/states/subgoals or not while running simulator"
            )
        },
    )
    pause: Optional[float] = field(
        default=0.5,
        metadata={
            "help": (
                "Pause time. (s)"
            )
        },
    )
    render: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Render img"
            )
        },
    )

@dataclass
class RolloutArguments:
    inference: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "inference or rollout"
            )
        },
    )
    verbose: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "whether to print statements during rollout"
            )
        },
    )
    n_traj_rollout: Optional[int] = field(
        default=10,
        metadata={
            "help": (
                "Number of ground truth traj we want to rollout."
            )
        },
    )
    id: Optional[str] = field(
        default="r-get-night-snack-h-aware-get-night-snack",
        metadata={
            "help": (
                "Policy model checkpoint."
            )
        },
    ) 
    traj_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Policy model checkpoint."
            )
        },
    )
    from_step_count: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "Step count to rollout from."
            )
        },
    )
    rollout_depth: Optional[int] = field(
        default=15,
        metadata={
            "help": (
                "Rollout depth."
            )
        },
    )
    policy_model_checkpoint: Optional[str] = field(
        default="/mnt/fs2/jphilipp/research-projects/checkpoints/experiment_0/checkpoint.ckpt",
        metadata={
            "help": (
                "Policy model checkpoint."
            )
        },
    )
    policy_model_checkpoint_short: Optional[str] = field(
        default="checkpoint.ckpt",
        metadata={
            "help": (
                "Policy model checkpoint."
            )
        },
    )
    rollout_dir: Optional[str] = field(
        default="/vision/u/emilyjin/marple_long/rollouts",
        metadata={
            "help": (
                "Rollout data directory."
            )
        },
    )
    inference_type: Optional[str] = field(
        default="sample",
        metadata={
            "help": (
                "Inference type: deterministic, beam_search"
            )
        },
    )
    temp: Optional[float] = field(
        default=1,
        metadata={
            "help": (
                "Temperature for softmax sampling."
            )
        },
    )
    num_samples: Optional[int] = field(
        default=100,
        metadata={
            "help": (
                "Number of rollouts to sample."
            )
        },
    )  
    extra_steps: Optional[int] = field(
        default=5,
        metadata={
            "help": (
                "Number of extra steps to sample."
            )
        },
    )        
    n_jobs: Optional[int] = field(
        default=20,
        metadata={
            "help": (
                "n jobs for parallel rollout."
            )
        },
    )
    rollout_level: Optional[str] = field(
        default="mid",
        metadata={
            "help": (
                "Inference type: mid, low"
            )
        },
    ) 
    simulator: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                'simulator goal conditioned rollout'
            )
        },
    )
    audio: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                'agent a mission'
            )
        },
    )
    a_mission: Optional[str] = field(
        default='change_outfit',
        metadata={
            "help": (
                'agent a mission'
            )
        },
    )
    b_mission: Optional[str] = field(
        default='move_plant_at_night',
        metadata={
            "help": (
                'agent b mission'
            )
        },
    ) 
    a_pref: Optional[str] = field(
        default=1,
        metadata={
            "help": (
                'agent a preference for the mission they are doing'
            )
        },
    )
    b_pref: Optional[str] = field(
        default=1,
        metadata={
            "help": (
                'agent b preference for the mission they are doing'
            )
        },
    )  
    room_config: Optional[str] = field(
        default='init_config_change_outfit_move_plant_at_night_simple_1',
        metadata={
            "help": (
                'agent b preference for the mission they are doing'
            )
        },
    )            
@dataclass
class Arguments:
    model: ModelArguments = ModelArguments()
    data: DataArguments = DataArguments()
    dataloader: DataLoaderArguments = DataLoaderArguments()
    trainer: TrainerArguments = TrainerArguments()
    optimizer: OptimizerArguments = OptimizerArguments()
    wandb: WandBArguments = WandBArguments()
    experiment: ExperimentArguments = ExperimentArguments()
    simulator: SimulatorArguments = SimulatorArguments()
    rollout: RolloutArguments = RolloutArguments()
    
cs = ConfigStore.instance()
cs.store(name="base_config", node=Arguments)
