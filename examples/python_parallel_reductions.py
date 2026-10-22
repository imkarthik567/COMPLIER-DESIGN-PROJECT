"""Demonstrates every supported deterministic integer reduction."""
def positive_value(value):
    return value + 1

print(parallel_sum(positive_value, 0, 4))
print(parallel_min(positive_value, 0, 4))
print(parallel_max(positive_value, 0, 4))
print(parallel_product(positive_value, 0, 4))
print(parallel_count(positive_value, 0, 4))
print(parallel_any(positive_value, 0, 4))
print(parallel_all(positive_value, 0, 4))
print(parallel_bit_and(positive_value, 0, 4))
print(parallel_bit_or(positive_value, 0, 4))
print(parallel_bit_xor(positive_value, 0, 4))
print(parallel_gcd(positive_value, 0, 4))
print(parallel_lcm(positive_value, 0, 4))
print(parallel_average(positive_value, 0, 4))
print(parallel_argmin(positive_value, 0, 4))
print(parallel_argmax(positive_value, 0, 4))
