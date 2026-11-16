"""`python -m hack_rl.play <task> [mjlab play options]`: mjlab's player with our tasks registered."""
import hack_rl.tasks  # noqa: F401
from mjlab.scripts.play import main

if __name__ == "__main__":
    main()
