"""A simple implementation of pipeline-parallel MLP training with the GPipe
schedule in JAX.
Run with `./launch.sh "mlp_training.py"`.
To collect and save a profile with nsys:
    nsys profile \
        --output=profile_training.nsys-rep \
        --trace-fork-before-exec=true \
        --cuda-graph-trace=node \
        ./launch.sh "mlp_training.py"
"""

import sys
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P


NUM_MICROBATCHES = 8
MICROBATCH_SIZE = 1024
FEATURE_DIM = 8192


# Initialize jax.distributed using parameters passed via sys.argv.
PROC_ID = int(sys.argv[1])
NUM_PROCS = int(sys.argv[2])
NUM_DEVICES_PER_PROC = int(sys.argv[3])
local_device_ids = [
    NUM_DEVICES_PER_PROC * PROC_ID + i for i in range(NUM_DEVICES_PER_PROC)
]
jax.distributed.initialize(
    "localhost:1234",
    num_processes=NUM_PROCS,
    process_id=PROC_ID,
    local_device_ids=local_device_ids,
)


# Create the meshes for each stage. Stage i holds devices managed by process i.
def make_stage_mesh(process_id):
    stage_devices = jax.local_devices(process_index=process_id)
    return jax.make_mesh(
        (len(stage_devices),),
        ("data",),
        devices=stage_devices,
    )


stage_meshes = [make_stage_mesh(process_id) for process_id in range(NUM_PROCS)]


# Define the computations that will run on each stage.
@jax.jit
def stage_fwd(W, x):
    x = x @ W
    x = jax.nn.relu(x)
    return x


@partial(jax.jit, donate_argnames=("x", "grads_acc"))
def stage_bwd(W, x, out_grads, grads_acc):
    _, bwd_fn = jax.vjp(stage_fwd, W, x)
    W_grad, x_grad = bwd_fn(out_grads)
    grads_acc = grads_acc + W_grad
    return grads_acc, x_grad


def final_fwd_and_loss(W, x, y):
    x = x @ W
    return jnp.mean((x - y) ** 2)


@partial(jax.jit, donate_argnames="accumulated_loss")
def final_stage_fwd(W, accumulated_loss, x, y):
    microbatch_loss = final_fwd_and_loss(W, x, y)
    accumulated_loss = accumulated_loss + microbatch_loss
    return accumulated_loss


@partial(jax.jit, donate_argnames=("x", "grads_acc"))
def final_stage_bwd(W, x, y, grads_acc):
    loss, bwd_fn = jax.vjp(final_fwd_and_loss, W, x, y)
    W_grad, x_grad, _ = bwd_fn(jnp.ones_like(loss))
    grads_acc = grads_acc + W_grad
    return grads_acc, x_grad


@partial(jax.jit, static_argnames=("lr",), donate_argnames=("W"))
def update_stage(W, W_grad, lr=0.1):
    return W - lr * (W_grad / NUM_MICROBATCHES)


# Define the MPMD program which executes each stage and coordinates data
# transfers using jax.device_put.
def train_step(microbatches_x, microbatches_y, stage_params):
    num_stages = len(stage_params)
    num_microbatches = len(microbatches_x)
    assert len(microbatches_y) == num_microbatches

    # Create a list of tasks sorted according to the order with which we will
    # enqueue work. This ordering follows the GPipe schedule.
    tasks = [
        (microbatch_idx, stage_idx, is_fwd)
        for stage_idx in range(num_stages)
        for microbatch_idx in range(num_microbatches)
        for is_fwd in (False, True)
    ]

    def task_key(task):
        microbatch_idx, stage_idx, is_bwd = task
        if is_bwd:
            stage_idx = -stage_idx
        return (is_bwd, microbatch_idx + stage_idx, stage_idx)

    tasks.sort(key=task_key)

    # Tracks the available inputs for each stage.
    fwd_inputs = {
        (microbatch_idx, 0): microbatch_x
        for microbatch_idx, microbatch_x in enumerate(microbatches_x)
    }
    bwd_inputs = {
        (microbatch_idx, num_stages - 1): 1.0
        for microbatch_idx, microbatch_x in enumerate(microbatches_x)
    }
    accumulated_loss = 0.0
    grads_by_stage = [jnp.zeros_like(p) for p in stage_params]

    # Main pipeline-parallel fwd/bwd loop.
    for microbatch_idx, stage_idx, is_bwd in tasks:
        curr_params = stage_params[stage_idx]
        is_final_stage = stage_idx == num_stages - 1

        # Forward pass of non-final stage.
        if not is_bwd and not is_final_stage:
            activation = stage_fwd(
                curr_params,
                fwd_inputs[(microbatch_idx, stage_idx)],
            )
            fwd_inputs[(microbatch_idx, stage_idx + 1)] = jax.device_put(
                activation,
                device=NamedSharding(stage_meshes[stage_idx + 1], P()),
            )

        # Forward pass of final stage.
        elif not is_bwd and is_final_stage:
            accumulated_loss = final_stage_fwd(
                curr_params,
                accumulated_loss,
                x=fwd_inputs[(microbatch_idx, stage_idx)],
                y=microbatches_y[microbatch_idx],
            )

        # Backward pass of final stage.
        elif is_bwd and is_final_stage:
            grads_by_stage[stage_idx], in_grads = final_stage_bwd(
                curr_params,
                x=fwd_inputs.pop((microbatch_idx, stage_idx)),
                y=microbatches_y[microbatch_idx],
                grads_acc=grads_by_stage[stage_idx],
            )
            bwd_inputs[(microbatch_idx, stage_idx - 1)] = jax.device_put(
                in_grads,
                device=NamedSharding(stage_meshes[stage_idx - 1], P()),
            )

        # Backward pass of non-final stage.
        else:
            grads_by_stage[stage_idx], in_grads = stage_bwd(
                curr_params,
                x=fwd_inputs.pop((microbatch_idx, stage_idx)),
                out_grads=bwd_inputs.pop((microbatch_idx, stage_idx)),
                grads_acc=grads_by_stage[stage_idx],
            )
            if stage_idx != 0:
                bwd_inputs[(microbatch_idx, stage_idx - 1)] = jax.device_put(
                    in_grads,
                    device=NamedSharding(stage_meshes[stage_idx - 1], P()),
                )

    # Update steps.
    for stage_idx in range(num_stages):
        stage_params[stage_idx] = update_stage(
            stage_params[stage_idx],
            grads_by_stage[stage_idx],
        )

    return (accumulated_loss / NUM_MICROBATCHES), stage_params


# Initialize inputs and run.
def make_inputs(rng_key, num_layers):
    x_key, *layer_keys = jax.random.split(rng_key, num_layers + 1)

    # Initialize x and y so that we try to learn the identity function.
    raw_data = jax.random.normal(
        x_key, (NUM_MICROBATCHES, MICROBATCH_SIZE, FEATURE_DIM)
    )
    microbatches_x = [
        jax.device_put(
            raw_data[microbatch_idx, ...],
            NamedSharding(stage_meshes[0], P()),
        )
        for microbatch_idx in range(NUM_MICROBATCHES)
    ]
    microbatches_y = [
        jax.device_put(
            raw_data[microbatch_idx, ...],
            NamedSharding(stage_meshes[-1], P()),
        )
        for microbatch_idx in range(NUM_MICROBATCHES)
    ]

    # Initialize weights with shape (FEATURE_DIM, FEATURE_DIM).
    weight_initializer = jax.nn.initializers.he_normal()
    layer_weights = [
        jax.device_put(
            weight_initializer(layer_key, (FEATURE_DIM, FEATURE_DIM)),
            NamedSharding(stage_meshes[layer_idx], P()),
        )
        for layer_idx, layer_key in enumerate(layer_keys)
    ]

    return microbatches_x, microbatches_y, layer_weights


def training_loop(num_steps=10):
    microbatches_x, microbatches_y, stage_params = make_inputs(
        jax.random.key(0), num_layers=4
    )
    microbatches_x = [[m.copy() for m in microbatches_x] for _ in range(num_steps)]
    microbatches_y = [[m.copy() for m in microbatches_y] for _ in range(num_steps)]
    jax.experimental.multihost_utils.sync_global_devices("start")

    for i in range(num_steps):
        print(f"===== STEP {i} {PROC_ID=} =====")

        loss, stage_params = train_step(
            microbatches_x[i],
            microbatches_y[i],
            stage_params,
        )
        if PROC_ID == NUM_PROCS - 1:
            print(f"Loss = {loss}")

        jax.experimental.multihost_utils.sync_global_devices(f"finish_step_{i}")


if __name__ == "__main__":
    training_loop()
