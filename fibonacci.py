def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

# The task asks for the 8th Fibonacci number. Assuming 0-indexing (F(0)=0, F(1)=1, ..., F(8)=21).
N = 8
result = fibonacci(N)
print(result)