name = input("Please enter your name: ")
surname = input("Please enter your surname: ")
age = int(input("Type your age: "))

magic_pill = 10
age -= magic_pill
message_name = "Hello " + name + " " + surname
message_age = "You are " + str(age) + " years old!"

message = message_name + message_age
print(message)

print(type(message))
print(type(age))
print(type(5))
print(type(4.678))
print(type(True))