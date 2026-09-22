import ttnn


def add_relu(a, b):
    out = ttnn.add(a, b)
    return ttnn.relu(out)
