import numpy as np
import tempfile
import jax
from flax import nnx
from orbax.checkpoint import v1 as ocp
from orbax.checkpoint import PyTreeCheckpointer
from pathlib import Path


class BestModelManager:
    """track best models across seed runs using orbax checkpointing. if base_dir is None, tmpdir is used"""

    def __init__(self, n_models: int, base_dir: Path | str | None = None):
        self.best_losses = np.full(n_models, np.inf)
        self._tmpdir = None
        if base_dir is None:
            self._tmpdir = tempfile.TemporaryDirectory()
            self.base_dir = Path(self._tmpdir.name)
        else:
            self.base_dir = Path(base_dir)
        self.checkpointer = PyTreeCheckpointer()
        print("Checkpointer: using dir", self.base_dir)

    def __del__(self):
        if self._tmpdir is not None:
            self._tmpdir.cleanup()

    def update_and_ckpt(self, models, metrics, metric_key):
        mets = np.array([m.compute()[metric_key] for m in metrics])
        model_state = nnx.state(models)
        for i in range(len(self.best_losses)):
            if self.best_losses[i] > mets[i]:
                self.best_losses[i] = mets[i]
                state = jax.tree.map(lambda x: x[i], model_state)
                self.checkpointer.save(self.base_dir / str(i), state, force=True)

    def load_best_model_states(self):
        states = []
        for i in range(len(self.best_losses)):
            state = self.checkpointer.restore(self.base_dir / str(i))
            states.append(state)
        return states

    def load_best_models(self, abstract_model):
        graphdef, _ = nnx.split(abstract_model)
        models = [nnx.merge(graphdef, st) for st in self.load_best_model_states()]
        return models
