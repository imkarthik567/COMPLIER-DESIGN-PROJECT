"""Recursive function calls and conditional returns."""
def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n - 1)

print(factorial(5))
print(factorial(10))
