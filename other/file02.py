def copy_files(file1, file2):
    with open(file2, "r") as f2:
        content = f2.read()
    with open(file1, "w") as f1:
        f1.write(content)
    return f1

copy_files("temp.txt", "temp2.txt")