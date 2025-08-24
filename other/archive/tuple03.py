N = 6
cnt = 0
if N <= 1:
    print(f"{N} is NOT a prime number")
else:
    for i in range(1, N - 1):
        if N % i == 0:
            cnt += 1

    if cnt > 1:
        print(f"{N} is NOT a prime number")
    else:
        print(f"{N} is a prime number")