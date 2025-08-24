string = "With that in mind, the Oval Office meeting between Donald Trump and Volodymyr Zelensky clearly got off on far more collegial — and presidential — footing today after the Ukrainian leader presented a letter from his wife for Trump to give to Melania Trump."

my_list = list(string)

dictionary = {}

for char in my_list:
    if char in dictionary:
        dictionary[char] += 1
    else:
        dictionary[char] = 1

max_value = max(list(dictionary.values()))

for key,value in dictionary.items():
    if value == max_value:
        if key == " ":
            print(f"Max value is {value} of the key space")
        else:
            print(f"Max value is {value} of the key {key}")