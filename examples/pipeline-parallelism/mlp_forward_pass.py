"""A simple implementation of a pipelined MLP forward pass in JAX.
Run with `./launch.sh`.
To collect and save a profile with nsys:
    nsys profile \
        --output=profile.nsys-rep \
        --trace-fork-before-exec=true \
        --python-sampling=true \
        --python-sampling-freq=100 \
        ./launch.sh
"""

import sys

import jax
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


# Define the computations that will run on each stage. Each stage executes one
# layer of the MLP. The last stage does not apply the activation function.
def make_stage_fn(stage_idx, num_stages):
    def stage_fwd(x, W):
        x = x @ W
        if stage_idx != num_stages - 1:
            x = jax.nn.relu(x)
        return x

    return jax.jit(stage_fwd)


stage_fns = [make_stage_fn(stage_idx, NUM_PROCS) for stage_idx in range(NUM_PROCS)]


def mlp_fwd(microbatches, stage_params):
    results = []
    for microbatch in microbatches:
        for stage_mesh, stage_fn, W in zip(stage_meshes, stage_fns, stage_params):
            curr_microbatch_sharding = NamedSharding(
                stage_mesh,
                microbatch.sharding.spec,
            )
            microbatch = jax.device_put(microbatch, curr_microbatch_sharding)
            microbatch = stage_fn(microbatch, W)
        results.append(microbatch)
    return results


# Initialize inputs and run.
def initialize_model_params(rng_key, num_layers):
    layer_keys = jax.random.split(rng_key, num_layers)
    weight_initializer = jax.nn.initializers.he_normal()
    layer_weights = [
        jax.device_put(
            weight_initializer(layer_key, (FEATURE_DIM, FEATURE_DIM)),
            NamedSharding(stage_meshes[layer_idx], P()),
        )
        for layer_idx, layer_key in enumerate(layer_keys)
    ]
    return layer_weights


def initialize_microbatches(rng_key):
    x = jax.device_put(
        jax.random.normal(rng_key, (MICROBATCH_SIZE, FEATURE_DIM)),
        NamedSharding(stage_meshes[0], P()),
    )
    # We reuse the same 'x' for every microbatch for simplicity.
    microbatches = [x for _ in range(NUM_MICROBATCHES)]
    return microbatches


def test_fwd_pass():
    params_rng_key, data_rng_key = jax.random.split(jax.random.key(0), 2)
    model_params = initialize_model_params(params_rng_key, num_layers=NUM_PROCS)
    microbatches = initialize_microbatches(data_rng_key)

    model_output = mlp_fwd(microbatches, model_params)
    jax.experimental.multihost_utils.sync_global_devices("fwd_pass_complete")

    print("Successfully ran forward pass!")


def profile_fwd_pass(num_steps=15):
    params_rng_key, data_rng_key = jax.random.split(jax.random.key(0), 2)
    model_params = initialize_model_params(params_rng_key, num_layers=NUM_PROCS)
    microbatches = initialize_microbatches(data_rng_key)

    jax.experimental.multihost_utils.sync_global_devices("start")

    for i in range(num_steps):
        print(f"===== STEP {i} {PROC_ID=} =====")
        out = mlp_fwd(microbatches, model_params)
        jax.experimental.multihost_utils.sync_global_devices(f"step_{i}_complete")


if __name__ == "__main__":
    # Uncomment the line below to run a single forward pass.
    # test_fwd_pass()

    # Uncomment the line below to run several forward passes back-to-back, e.g.
    # for profiling with nsys.
    profile_fwd_pass()

    # Uncomment the lines below to run forward passes back-to-back while saving
    # xprof profiles for each process.
    # with jax.profiler.trace(f"xprof_profile/proc_id={PROC_ID}"):
    #     profile_fwd_pass()

