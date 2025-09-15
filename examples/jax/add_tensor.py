import jax
from jax import jit

def random_input_tensor(shape, key=42, on_device=True):
    def random_input(shape, key):
        return jax.random.uniform(jax.random.PRNGKey(key), shape=shape)

    jitted_tensor_creator = jax.jit(random_input, static_argnums=[0, 1], backend="cpu")
    tensor = jitted_tensor_creator(shape, key)
    if on_device:
        tensor = jax.device_put(tensor, jax.devices()[0])
    return tensor


def add_tensors():

    # Create random input tensors
    tensor1 = random_input_tensor((3, 3), key=0, on_device=True)
    tensor2 = random_input_tensor((3, 3), key=1, on_device=True)

    # Define a simple addition function
    def add_fn(x, y):
        return x + y

    # JIT compile the addition function
    jitted_add_fn = jit(add_fn)

    # Perform the addition
    result = jitted_add_fn(tensor1, tensor2)

    print("Result of addition:", result)

if __name__ == "__main__":
    add_tensors()