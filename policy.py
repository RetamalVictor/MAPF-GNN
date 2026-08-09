"""How action logits become actions.

Taking the argmax makes the policy deterministic, and the environment already is.
Once a joint state repeats, it repeats forever - which is what a deadlock here
actually is. Measured on 200 episodes at 20 robots, 98% of failed episodes were
completely frozen in a single repeating state and the remaining 2% were short
cycles; none were anything else. That is also why extending the evaluation
horizon barely moved the numbers: no horizon escapes a fixed point.

Sampling breaks the symmetry that creates those fixed points. It is not a
substitute for a good policy - uniform random actions score zero - but it lets a
good policy escape a standoff two robots would otherwise hold forever.

author: Victor Retamal
"""
import torch

# Sampling temperature that performed best across fleet sizes on this task.
# Below it, symmetry breaks too rarely; far above it the policy's own preference
# is drowned out and success collapses.
DEFAULT_TEMPERATURE = 3.0


def select_actions(logits, temperature=0.0, generator=None):
    """Pick one action per robot.

    Args:
        logits: (episodes, robots, actions)
        temperature: 0 for argmax, otherwise sample from softmax(logits / T)
        generator: optional torch.Generator, for reproducible sampling
    """
    if temperature <= 0:
        return logits.argmax(dim=2)

    episodes, robots, actions = logits.shape
    probabilities = torch.softmax(logits / temperature, dim=2)
    sampled = torch.multinomial(
        probabilities.reshape(-1, actions), 1, generator=generator
    )
    return sampled.reshape(episodes, robots)


def add_temperature_argument(parser, default=0.0):
    parser.add_argument(
        "--temperature", type=float, default=default,
        help="Sampling temperature for action selection. 0 takes the argmax, "
             f"which is deterministic and lets deadlocks persist forever; "
             f"around {DEFAULT_TEMPERATURE} breaks that symmetry and raises the "
             "success rate substantially.",
    )
    return parser
