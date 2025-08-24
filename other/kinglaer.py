with open("king_laer.txt", "r") as f:
    lines = f.readlines()

for i in range(len(lines)):
    if lines[i].isupper():
        lines[i] = f"\n{lines[i]}\n"
    else:
        lines[i] = f"\t{lines[i]}"

with open("king_laer.txt", "w") as f:
    for line in lines:
        f.write(line)