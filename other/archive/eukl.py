def euk(x, y):
    if x == y:
        return x
    if x < y:
        return euk(x, y - x)
    else:
        return euk(x - y, x)

print(euk(255, 155))
