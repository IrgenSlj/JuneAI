my_set = {numb + 1 for numb in range(10)}

print(f"{my_set}\n")

my_set = {numb for numb in range(10) if numb % 2 == 0}

print(f"{my_set}\n")

my_set = {numb if numb % 2 == 0 else numb *100 
          for numb in range(10)}

print(f"{my_set}\n")

my_set = {(i, j) for i in range(10)
          for j in range(10)}

print(f"{my_set}\n")

my_set = {(i, j) for i in range(10) if i % 2 == 0
          for j in range(10) if j % 2 != 0}

print(f"{my_set}\n")
