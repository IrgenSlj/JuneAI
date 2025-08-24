age = int(input("What's your age? "))

if age > 65:
    print("You are a pensionist")
elif age >= 18:
    print("You are an adult")
else:
    print("You are underaged")

print("\n0-----------------------------0")

pl_a_01 = int(input("\nPlayer 1, throw your first dice \n"))
pl_a_02 = int(input("Now, throw your second dice \n"))

pl_b_01 = int(input("Player 2, throw your first dice \n"))
pl_b_02 = int(input("Now, throw your second dice \n"))

pl_a_sum = pl_a_01 + pl_a_02
pl_b_sum = pl_b_01 + pl_b_02

if pl_a_sum == pl_b_sum:
    print("It's a draw")
elif pl_a_sum > pl_b_sum:
    print("Player 1 wins")
else:
    print("Player 2 wins")