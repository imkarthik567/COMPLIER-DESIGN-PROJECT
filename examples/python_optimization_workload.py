"""A small scoring workload that demonstrates several safe optimizations."""

# These constant and unused calculations are deliberately included so the
# compiler can demonstrate folding, copy simplification, and dead-code removal.
points_per_square = 2 * 3
unused_limit = 100 * 100
unused_copy = unused_limit

score = 0
for value in range(1, 21):
    square = value * value
    weighted = square * points_per_square
    adjusted = weighted + 0
    score += adjusted

print(score)
