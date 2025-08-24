i = 1
while i <= 10:
    print("#")
    i += 1

cnt = 9 

while cnt >= 1:
    print(cnt)
    cnt += 2



number = int(input("Enter a number 0-9: "))

while number < 0 or number > 9:
    number = int(input("Enter a number 0-9: "))

print("You entered: " + str(number))


active = True

while active:
    u_input = input("Type text or 'quit': ")
    if u_input == "quit":
        active = False
    else:
        print("You printed: " + u_input)