A = {1, 2, 3, 4}
B = {4, 5}

print("union: " + str(A | B))
print(f"union: {A.union(B)}")
print()

print("intersection: " + str(A & B))
print(f"intersection: {A.intersection(B)}")
print()

print("difference: " + str(B - A))
print(f"difference: {A.difference(B)}")
print()

print("symmetric difference: " + str(A ^ B))
print(f"symmetric difference: {A.symmetric_difference(B)}")
print()

print(f"{B.issubset(A)}")