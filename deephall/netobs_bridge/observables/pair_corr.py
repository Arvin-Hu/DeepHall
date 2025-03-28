# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

from jax import numpy as jnp
from jax.numpy import cos, sin
from netobs.observables import Estimator, Observable

from deephall.netobs_bridge.hall_system import HallSystem


class PairCorrelation(Observable):
    def shapeof(self, system) -> tuple[int, ...]:
        return ()


class PairCorrelationEstimator(Estimator[HallSystem]):
    observable_type = PairCorrelation

    def __init__(self, adaptor, system, estimator_options, observable_options):
        super().__init__(adaptor, system, estimator_options, observable_options)
        self.bins = self.options.get("bins", 200)

    def empty_val_state(
        self, steps: int
    ) -> tuple[dict[str, jnp.ndarray], dict[str, Any]]:
        del steps
        return {}, {"pair_corr": jnp.zeros(self.bins)}

    def evaluate(
        self, i, params, key, data, system, state, aux_data
    ) -> tuple[dict[str, jnp.ndarray], dict[str, Any]]:
        del i, params, aux_data, key, system
        data = jnp.reshape(data, (-1, *data.shape[-2:]))
        batch_size, nelec, _ = data.shape
        theta, phi = data[..., 0], data[..., 1]
        xyz_data = jnp.stack(
            [sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)], axis=-1
        )
        cos12 = jnp.sum(xyz_data[..., :, None, :] * xyz_data[..., None, :, :], axis=-1)
        theta12 = jnp.arccos(cos12[:, *jnp.triu_indices(nelec, 1)].reshape(-1))
        to_add, _ = jnp.histogram(
            theta12, self.bins, (0, jnp.pi), weights=1 / sin(theta12)
        )
        # Norm factor about evaluation steps is not divided. Remember to do it yourself.
        # An additional 2 comes from (i != j) => (i < j)
        state["pair_corr"] += to_add * 4 * self.bins / batch_size / nelec**2 / jnp.pi
        return {}, state

    def digest(self, all_values, state) -> dict[str, jnp.ndarray]:
        del all_values, state
        return {}


DEFAULT = PairCorrelationEstimator  # Useful in CLI
