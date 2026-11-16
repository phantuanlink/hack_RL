"""`python -m hack_rl.train <task> [mjlab train options]`: mjlab's trainer with our tasks registered."""
import hack_rl.tasks  # noqa: F401  (registers Hack-* tasks)
from mjlab.scripts.train import main


def _wandb_compat() -> None:
    """RSL-RL's wandb writer passes Settings(start_method=...), which current wandb rejects. Drop it."""
    try:
        import wandb
    except ImportError:
        return
    orig = wandb.Settings
    if getattr(orig, "_hack_rl_compat", False):
        return

    def settings(*args, **kwargs):
        kwargs.pop("start_method", None)
        return orig(*args, **kwargs)

    settings._hack_rl_compat = True
    wandb.Settings = settings


_wandb_compat()


if __name__ == "__main__":
    main()
