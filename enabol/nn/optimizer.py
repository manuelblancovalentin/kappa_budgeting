from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import tensorflow as tf

from ..utils import flatten_tensors


@dataclass(frozen=True)
class OptimizerStep:
    """Raw optimizer proposal before controller intervention."""

    updates: list[tf.Tensor]
    update_flat: tf.Tensor
    learning_rate: tf.Tensor


class BaseUpdateRule(ABC):
    """Base class for optimizer-style update rules.

    The update rule owns optimizer state and proposes a raw parameter delta.
    Controllers then decide how to modify that raw delta before it is applied.
    """

    optimizer_id = "OPT-BASE"

    def reset(self) -> None:
        """Reset optimizer state before a fresh run."""

    def describe(self) -> str:
        return f"{self.optimizer_id}()"

    def __repr__(self) -> str:
        return self.describe()

    @abstractmethod
    def propose(
        self,
        grads: list[tf.Tensor],
        variables: list[tf.Variable],
    ) -> OptimizerStep:
        pass


class SGDUpdateRule(BaseUpdateRule):
    """Plain SGD update proposal: delta = -eta * grad."""

    optimizer_id = "OPT-SGD"

    def __init__(self, learning_rate: float | tf.Tensor):
        self.learning_rate = tf.cast(learning_rate, tf.float32)

    def describe(self) -> str:
        return "\n".join(
            [
                f"{self.optimizer_id}(",
                f"  learning_rate: {float(self.learning_rate.numpy()):.6g}",
                ")",
            ]
        )

    def propose(
        self,
        grads: list[tf.Tensor],
        variables: list[tf.Variable],
    ) -> OptimizerStep:
        updates = [-self.learning_rate * tf.cast(grad, tf.float32) for grad in grads]
        return OptimizerStep(
            updates=updates,
            update_flat=flatten_tensors(updates),
            learning_rate=self.learning_rate,
        )
