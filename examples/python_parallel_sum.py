"""Real process-level parallel execution of a pure worker over independent inputs."""
def square(value):
    return value * value

total = parallel_sum(square, 0, 1000)
print(total)
