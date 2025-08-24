'''
hid_num = 66
num = hid_num + 1
tries = 0
print("Welcome to the guess the number game, you have 10 tries!")

while hid_num != num:
    for i in range(10):    
        num = int(input("Guess the number? (or type 0 to quit)"))

        if num == hid_num or num == 0:
            print("That's it! you won!")
            break
        elif num < hid_num:
            print("Go higher")
        else:
            print("Go lower!")
        
    else:
        print("Game over!")
        break

'''
hidden = 66
cnt = 0
guess = int(input("Type a number to guess hidden one: "))

while True:
    cnt += 1

    if cnt == 10:
        print("Game over! You loose")
        break
    elif guess == hidden:
        print("You found it! Congrats!")
        break
    elif guess < hidden:
        print("Go higher!")
    else:
        print("Go lower!")

    guess = int(input("Try another number: "))