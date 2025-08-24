hidden = [1, 3, 2, 1, 4, 4, 3, 2]
numbers = [1, 2, 3, 4, 5, 6, 7, 8]

N = 8

state = ["O", "O", "O", "O", "O", "O", "O", "O"]
# other states: open, temp_open

while True:
    # read first position
    pos1 = int(input(f"\nType first card position (1 - {N}): "))
    while pos1 < 1 or pos1 > N or state[pos1 - 1] == "X":
        pos1 = int(input(f"Error! Type first card position (1 - {N}): "))
    # read second position
    pos2 = int(input(f"Type second card position (1 - {N}): "))
    while pos2 < 1 or pos2 > N or state[pos2 - 1] == "X" or pos2 == pos1:
        pos2 = int(input(f"Error! Type second card position (1 - {N}): "))

    pos1 -= 1
    pos2 -= 1

    # change state: both temp open
    state[pos1] = "X"
    state[pos2] = "X"

    # print numbering
    print()
    print()
    print()
    for position in numbers:
        print(position, end=" ")

    # print current state
    print()
    for position in range(0, N):
        if state[position] == "O":
            print("#", end=" ")
        elif state[position] == "I":
            print(hidden[position], end=" ")
        else:
            print(hidden[position], end=" ")
    print()
    # check if same the open, else closed
    if hidden[pos1] == hidden[pos2]:
        state[pos1], state[pos2] = "I", "I"
    else:
        state[pos1] = "O"
        state[pos2] = "O"

    # print current state
    print()
    for position in range(0, N):
        if state[position] == "O":
            print("#", end=" ")
        elif state[position] == "I":
            print(hidden[position], end=" ")
        else:
            print(hidden[position], end=" ")
    print()

    # check ending
    if "O" not in state:
        print("\nYou won!")
        break
