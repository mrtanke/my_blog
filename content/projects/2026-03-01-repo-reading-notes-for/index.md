---
title: 'Repo Reading Notes for OpenPI'
date: 2026-02-28T20:49:01+00:00
draft: True
description: ''
tag: 'Posts'
ShowWordCount: true
ShowReadingTime: false
tags:
  - vision-language-action
  - robotics
  - embodied-intelligence
---


After reading the paper: [π0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164)), I decided to spend a few days walking through the official implementation, [openpi](https://github.com/Physical-Intelligence/openpi), to understand how everything work in practice.

There are several questions I want to find answer. On the big side: **how does this repo turn VLM features into robot actions, and how are training and inference actually wired together?** On the smaller side: **how is the two-expert MoE implemented, and how do observations influence the final action output?**

With these questions in mind, I started digging into the codebase, working through the **dataset pipeline**, **model definition**, and the **training inference flow** to map the paper’s ideas to concrete implementation details.

# Main Idea

# train_pytorch.py

There are three lines for `main()` function, let’s check them line by line.

```python
def main():
    init_logging() # -------------- 1
    config = _config.cli() # ------ 2
    train_loop(config) # ---------- 3
```

For the first line `init_logging()`, it actually comes from the function `def init_logging()`.

## init_logging()

This line is connected with a function `init_logging()`, whic sets up the logging pipeline by defining a custom formatter and attaching it to the root logger’s console handler (`StreamHandler`).

It also sets the root logger’s level to `INFO` (`logger.setLevel(logging.INFO)`), meaning **INFO and above** are emitted (`info`, `warning`, `error`, `critical`), while **DEBUG is filtered out**—so for `logging.debug(...)`, no `LogRecord` is emitted and the formatter never runs.

```python
def init_logging():
		"""
		Set up logging to the console with a custom format
		"""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record): # record: a LogRecord object
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    
    # get the root logger object
    **logger = logging.getLogger()**
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        ch = logging.StreamHandler() # create a handler object
        ch.setFormatter(formatter) # set the formatter for the handler to convert the log recored in the logger to the format we want
        logger.addHandler(ch) # attach to the logger
    else:
        logger.handlers[0].setFormatter(formatter)
```

The **pipeline** of logging from source code to output is:

1. The source code calls `logging.info("Saved checkpoint")`.
2. That call uses the **root logger** (the global logger has been configured by `init_logging()`).
3. The **logger checks its level**: if debug, then it stops here and does nothing.
4. The logger checks the message’s level against **its threshold** (`logger.setLevel(logging.INFO)`). If the message is below the threshold (e.g., DEBUG), it stops and does nothing.
5. If allowed, the logger creates a **LogRecord** for this message (an “envelope” containing message + metadata).
6. The logger sends the LogRecord to each attached **Handler** in `logger.handlers`.
7. Each handler uses its **Formatter** to convert `record → string`.
8. The handler outputs the string to its destination (console, file, etc.).

## config = _config.cli()

The second line, `config = _config.cli()`, calls into `openpi/training/config.py` to build a `TrainConfig` object from the command line arguments.

```python
import openpi.training.config as _config

config = _config.cli()  # ------ 2
```

In `config.py`, the repo defines many preset `TrainConfig`s (some intended for inference, some for training/fine-tuning). These presets are collected in `_CONFIGS`, then turned into a dictionary `_CONFIGS_DICT` that maps:

- `TrainConfig.name` → `TrainConfig` instance

so each preset can be selected by name.

When we run a command like:

```bash
python scripts/train_pytorch.py pi0_aloha
```

`tyro.extras.overridable_config_cli(...)` selects the preset named `"pi0_aloha"`, loads the corresponding `TrainConfig`, applies any CLI overrides if existing (e.g., `--exp_name`, `--batch_size`, `--resume`), and returns the final `config` object used by the rest of the script.

```python
def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli(
        {k: (k, v) for k, v in **_CONFIGS_DICT**.items()}
    )

**_CONFIGS_DICT** = {config.name: config for config in _CONFIGS}

**_CONFIGS** = [
    **TrainConfig**(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    **TrainConfig**(...),
    **TrainConfig**(...),
    ...
]
```

We can take a specific `TrainConfig`: `pi0_aloha`, which is an inference-oriented ALOHA preset. This preset defines four key pieces:

- `name`: the preset identifier used on the CLI
- `model`: the π-model architecture configuration (action horizon/dimension, max token length, backbone variants, etc).
- `data`: the **data pipeline configuration**, it specifies the transform chain around the model including **repack transforms, data transforms, normalization and model transforms.**
- `policy_metadata`: extra runtime hints for the policy (e.g., a safe reset pose)

```python
TrainConfig(
    name="pi0_aloha",
    model=pi0_config.Pi0Config(),
    data=**LeRobotAlohaDataConfig**(
        assets=AssetsConfig(asset_id="trossen"),
    ),
    policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
)
```

Here, `assets` means when building the data pipeline, load assets with `"trossen"` asset_id. The asset includes normalization stats (mean and std or quantiles) for state and action normalization.

`reset_pose` is meant as an initial arm configuration. It is 6 values because it typically refers to **arm joints only**, with the gripper handled separately.

### class LeRobotAlohaDataConfig

Next see more detials about `LeRobotAlohaDataConfig` is a `DataConfigFactory`: it **builds a `DataConfig`** (the actual pipeline description) for ALOHA-style data.

It starts with three knobs:

- `use_delta_joint_actions=True`: convert joint action dimensions into deltas relative to current state, and keep gripper absolute
- `default_prompt`: if provided, inject it when `"prompt"` is missing
- `adapt_to_pi=True`: convert from “standard ALOHA conventions” into PI’s internal conventions used when training the base model

Then it defines the transform layers:

- `repack_transforms`: **key renaming**, so dataset samples match the common runtime key format (e.g., mapping `images.cam_high → observation.images.top`)
- `data_transforms`: **robot/platform adapter logic** that changes the **semantics** to match what the PI’s canonical robot representation expects. For ALOHA this includes:
    - packing/unpacking ALOHA observations and actions via `AlohaInputs` / `AlohaOutputs`
    - optional conversion between ALOHA conventions and PI’s internal conventions (`adapt_to_pi`)
    - converting **action** representations **delta ↔ absolute**, using `DeltaActions` and `AbsoluteActions`.
- `model_transforms`: **model-specific preprocessing** independent of the specific robot platform. For PI0/PI05 this typically includes:
    - injecting a default prompt if missing
    - resizing images to the fixed input size
    - tokenizing the prompt with the model’s tokenizer and enforcing max token length
    - padding state/action vectors to fixed dimensions so batching works consistently.

Finally, its `create(...)` method returns a concrete `DataConfig` that can be used by both the training dataloader and the inference pipeline.

```python
@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    adapt_to_pi: bool = True

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )
```

Researchers often prefer **Delta Actions** when training AI models (like Gemini or specialized robot brains) because it mimics how humans move. We don't think "Move hand to coordinate (22, 14, 5)"; we think "Move hand a little further toward the coffee cup." It makes the learning process more flexible and less prone to "teleporting" errors where a robot tries to move instantly to a far-off position. ([Eßer, et al. 2025](https://openreview.net/pdf?id=GGuNkjQSrk))

### train_loop(config)

The first two lines are for pre-setting, and the third line is the main content for training.