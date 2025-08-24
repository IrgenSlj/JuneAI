from random import randrange

words = ["hello world", "good party", "automotive", "grassroots"]

hidden_word = words[randrange(0, len(words))]
guessed_letter = ["_" for _ in range(len(hidden_word))] 
guessed_word = ""

print(f"Guess hidden word: ")
for i in range(len(guessed_letter)):
    print(f"{guessed_letter[i]} ", end="")
print()

while True:
    cnt = 0

    if guessed_word == hidden_word:
        print(f"\nYou won! The hidden word was: {guessed_word}")
        break
    else:
        char = input("\nEnter a letter: ")
        for i in range(len(hidden_word)):
            if char == hidden_word[i]:
                cnt += 1
                guessed_letter[i] = char
            else:
                pass
        print(f"\n{char} is {cnt} times in the hidden word")
        print()

    guessed_word = ""

    print(f"Current hidden word: ")
    for i in range(len(guessed_letter)):
        print(f"{guessed_letter[i]} ", end="")
        guessed_word = f"{guessed_word}{guessed_letter[i]}"

