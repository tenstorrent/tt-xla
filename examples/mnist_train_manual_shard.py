# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
from jax.experimental import shard_map
import jax.lax as lax
import torchvision
import numpy as np


class NetConfig:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size


class TrainingConfig:
    def __init__(self, num_epochs=10, batch_size=128, learning_rate=0.001):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate


class LoggerConfig:
    def __init__(self, log_every_n_steps=10):
        self.log_every_n_steps = log_every_n_steps


class ExperimentConfig:
    def __init__(self, net_config=None, training_config=None, logger_config=None):
        self.net_config = net_config or NetConfig()
        self.training_config = training_config or TrainingConfig()
        self.logger_config = logger_config or LoggerConfig()


def init_device():
    # placeholder for TT device
    pass


def export_to_stablehlo(forward_fn, params, *args):
    fwd_jit = jax.jit(forward_fn)
    fwd_lowered = fwd_jit.lower(params, *args)

    fwd_stablehlo = fwd_lowered.compile()
    print(fwd_lowered.as_text())


def load_mnist():
    mnist = {
        "train": torchvision.datasets.MNIST("./data", train=True, download=True),
        "test": torchvision.datasets.MNIST("./data", train=False, download=True),
    }

    ds = {}

    for split in ["train", "test"]:
        ds[split] = {
            "image": mnist[split].data.numpy(),
            "label": mnist[split].targets.numpy(),
        }

        ds[split]["image"] = jnp.float32(ds[split]["image"]) / 255

        ds[split]["label"] = jnp.int16(ds[split]["label"])

        ds[split]["image"] = ds[split]["image"].reshape(-1, 28 * 28)

    train_images, train_labels = ds["train"]["image"], ds["train"]["label"]
    test_images, test_labels = ds["test"]["image"], ds["test"]["label"]

    train_labels = jax.nn.one_hot(train_labels, 10).astype(jnp.float32)
    test_labels = jax.nn.one_hot(test_labels, 10).astype(jnp.float32)

    perm = jax.random.permutation(jax.random.PRNGKey(0), len(train_images))
    train_images, train_labels = train_images[perm], train_labels[perm]

    train_size = int(0.8 * len(train_images))
    val_size = len(train_images) - train_size

    train_images, val_images = (
        train_images[:train_size],
        train_images[train_size : train_size + val_size],
    )
    train_labels, val_labels = (
        train_labels[:train_size],
        train_labels[train_size : train_size + val_size],
    )

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


def mlp_model(params, x):
    w1, b1, w2, b2, w3, b3 = params
    h1 = jnp.maximum(jnp.dot(x, w1) + b1, 0.0)
    h2 = jnp.maximum(jnp.dot(h1, w2) + b2, 0.0)
    logits = jnp.dot(h2, w3) + b3
    return logits


def init_mlp_params(key, input_size, hidden_size, output_size):
    w1_shape = (input_size, hidden_size)
    b1_shape = (hidden_size,)
    w2_shape = (hidden_size, hidden_size)
    b2_shape = (hidden_size,)
    w3_shape = (hidden_size, output_size)
    b3_shape = (output_size,)

    w1 = random.normal(key, w1_shape) * jnp.sqrt(2.0 / w1_shape[0])
    w1 = w1.astype(jnp.float32)
    b1 = jnp.zeros(b1_shape, dtype=jnp.float32)
    w2 = random.normal(key, w2_shape) * jnp.sqrt(2.0 / w2_shape[0])
    w2 = w2.astype(jnp.float32)
    b2 = jnp.zeros(b2_shape, dtype=jnp.float32)
    w3 = random.normal(key, w3_shape) * jnp.sqrt(2.0 / w3_shape[0])
    w3 = w3.astype(jnp.float32)
    b3 = jnp.zeros(b3_shape, dtype=jnp.float32)

    return (w1, b1, w2, b2, w3, b3)


def mse_loss(logits, y):
    return jnp.mean((logits - y) ** 2)


def update(params, x_batch, y_batch, learning_rate):
    def loss_fn(p):
        logits = mlp_model(p, x_batch)
        return mse_loss(logits, y_batch)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    grads = jax.tree_util.tree_map(
        lambda g: lax.pmean(g, axis_name="dp") if g.ndim > 0 else g, grads
    )
    w1, b1, w2, b2, w3, b3 = params
    dw1, db1, dw2, db2, dw3, db3 = grads
    updated_params = (
        w1 - learning_rate * dw1,
        b1 - learning_rate * db1,
        w2 - learning_rate * dw2,
        b2 - learning_rate * db2,
        w3 - learning_rate * dw3,
        b3 - learning_rate * db3,
    )
    loss = lax.pmean(loss, axis_name="dp")
    return updated_params, loss


def validation_loss(params, x_batch, y_batch):
    def loss_fn(p):
        logits = mlp_model(p, x_batch)
        return mse_loss(logits, y_batch)

    loss = loss_fn(params)
    loss = lax.pmean(loss, axis_name="dp")

    return loss


def argmax_on_cpu(array):
    array_cpu = jax.device_put(array, jax.devices("cpu")[0])
    with jax.default_device(jax.devices("cpu")[0]):
        argmax_result = jnp.argmax(array_cpu, axis=-1)
        argmax_result = argmax_result.astype(jnp.uint32)
    return argmax_result


def compute_accuracy(logits, y):
    predictions = argmax_on_cpu(logits)
    true_labels = argmax_on_cpu(y)
    correct = jnp.mean(predictions == true_labels)
    return correct


def train_mlp(
    x_train_host,
    y_train_host,
    x_val_host,
    y_val_host,
    x_test_host,
    y_test_host,
    key,
    config,
    param_sharding,
):
    net_config = config.net_config
    logger_config = config.logger_config
    training_config = config.training_config
    batch_size = training_config.batch_size
    num_epochs = training_config.num_epochs

    input_size = net_config.input_size
    hidden_size = net_config.hidden_size
    output_size = net_config.output_size

    params = init_mlp_params(key, input_size, hidden_size, output_size)
    params = lax.with_sharding_constraint(params, param_sharding)

    num_batches = x_train_host.shape[0] // batch_size
    num_devices = jax.local_device_count()
    devices = jax.local_devices()
    mesh_devices = mesh_utils.create_device_mesh((num_devices,))
    mesh = Mesh(mesh_devices, ("dp",))
    data_sharding = NamedSharding(mesh, PartitionSpec("dp"))

    @jax.jit
    def training_step(params, x_batch, y_batch, lr):
        return shard_map.shard_map(
            lambda p, x, y, lr: update(p, x, y, lr),
            mesh=mesh,
            in_specs=(
                PartitionSpec(None),
                PartitionSpec("dp"),
                PartitionSpec("dp"),
                PartitionSpec(),
            ),
            out_specs=(PartitionSpec(None), PartitionSpec()),
        )(params, x_batch, y_batch, lr)

    for epoch in range(num_epochs):
        batch_loss_accum = 0.0
        batch_accuracy_accum = 0.0

        for i in range(num_batches):
            x_batch_host, y_batch_host = (
                x_train_host[i * batch_size : (i + 1) * batch_size],
                y_train_host[i * batch_size : (i + 1) * batch_size],
            )

            x_batch = jnp.array(x_batch_host)
            y_batch = jnp.array(y_batch_host)

            x_batch = lax.with_sharding_constraint(x_batch, data_sharding)
            y_batch = lax.with_sharding_constraint(y_batch, data_sharding)

            params, loss = training_step(
                params, x_batch, y_batch, training_config.learning_rate
            )

            batch_loss_accum += loss

            if (i + 1) % logger_config.log_every_n_steps == 0:
                avg_loss = batch_loss_accum / logger_config.log_every_n_steps
                print(f"Epoch {epoch}, Batch {i +1}, Loss: {avg_loss}")
                batch_loss_accum = 0.0

        val_loss = evaluate(params, x_val_host, y_val_host, mesh, data_sharding)
        print(f"Epoch {epoch}, Validation Loss: {val_loss}")

    test_loss, test_accuracy = evaluate(
        params, x_test_host, y_test_host, mesh, data_sharding
    )
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    return params


def evaluate(params, x_test, y_test, mesh, data_sharding, batch_size=16):
    total_loss = 0.0
    num_samples = 0

    num_devices = jax.local_device_count()
    devices = jax.local_devices()

    @jax.jit
    def validation_step(params, x_batch, y_batch):
        return shard_map.shard_map(
            lambda local_x, local_y: validation_loss(params, local_x, local_y),
            mesh=mesh,
            in_specs=(PartitionSpec("dp"), PartitionSpec("dp")),
            out_specs=PartitionSpec(None),
        )(x_batch, y_batch)

    x_batch = jnp.zeros((batch_size, x_test.shape[1]))
    y_batch = jnp.zeros((batch_size, y_test.shape[1]))

    for i in range(0, len(x_test), batch_size):
        current_batch_size = min(batch_size, len(x_test) - i)
        if current_batch_size < batch_size:
            x_batch_host = x_test[i : i + current_batch_size]
            y_batch_host = y_test[i : i + current_batch_size]
            x_batch = jnp.array(x_batch_host)
            y_batch = jnp.array(y_batch_host)
        else:
            x_batch = x_batch.at[:current_batch_size].set(
                x_test[i : i + current_batch_size]
            )
            y_batch = y_batch.at[:current_batch_size].set(
                y_test[i : i + current_batch_size]
            )

        x_batch = lax.with_sharding_constraint(x_batch, data_sharding)
        y_batch = lax.with_sharding_constraint(y_batch, data_sharding)

        loss = validation_step(params, x_batch, y_batch)

        total_loss += loss
        num_samples += current_batch_size

    avg_loss = total_loss / num_samples

    return avg_loss


def train_mnist():
    config = ExperimentConfig()
    training_config = config.training_config
    net_config = config.net_config
    logger_config = config.logger_config

    init_device()

    num_devices = jax.local_device_count()
    devices = jax.local_devices()
    mesh = Mesh(np.array(jax.local_devices()), axis_names=("dp",))

    data_sharding = NamedSharding(mesh, PartitionSpec("dp"))
    param_sharding = NamedSharding(mesh, PartitionSpec(None))

    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist()
    key = random.PRNGKey(0)

    train_mlp(
        x_train, y_train, x_val, y_val, x_test, y_test, key, config, param_sharding
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--stablehlo":
        key = random.PRNGKey(0)
        input_size = 784
        hidden_size = 128
        output_size = 10
        batch_size = 16
        learning_rate = 0.01

        params = init_mlp_params(key, input_size, hidden_size, output_size)

        x_batch = jnp.zeros((batch_size, input_size))
        y_batch = jnp.zeros((batch_size, output_size))

        num_devices = jax.local_device_count()
        devices = jax.local_devices()
        mesh = Mesh(np.array(jax.local_devices()), axis_names=("dp",))
        data_sharding = NamedSharding(mesh, PartitionSpec("dp"))

        x_batch = lax.with_sharding_constraint(x_batch, data_sharding)
        y_batch = lax.with_sharding_constraint(y_batch, data_sharding)

        @jax.jit
        def training_step(params, x_batch, y_batch, lr):
            return shard_map.shard_map(
                lambda p, x, y, lr: update(p, x, y, lr),
                mesh=mesh,
                in_specs=(
                    PartitionSpec(None),
                    PartitionSpec("dp"),
                    PartitionSpec("dp"),
                    PartitionSpec(),
                ),
                out_specs=(PartitionSpec(None), PartitionSpec()),
            )(params, x_batch, y_batch, lr)

        print("StableHLO representation of training_step function:")
        lowered = training_step.lower(params, x_batch, y_batch, learning_rate)
        print(lowered.as_text())
    else:
        train_mnist()
