def files(file1, file2, file3):
    with open(file1) as f1:
        c1 = f1.read()
    with open(file2) as f2:
        c2 = f2.read()
    with open(file3, "a") as f3:
        f3.write("\n" + str(c1) + "\n" + str(c2))

files("tempA.txt", "tempB.txt", "tempC.txt")