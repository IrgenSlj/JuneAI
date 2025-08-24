for a in range(0, 21):
    for b in range(0, 21):
        for c in range(0, 21):
            if pow(a, 2) + pow(b, 2) == pow(c, 2):
                print(f"{a}^2 + {b}^2 = {c}^2")
                print(f"{pow(a, 2)} + {pow(b, 2)} = {pow(c, 2)}")
                print("")
        
