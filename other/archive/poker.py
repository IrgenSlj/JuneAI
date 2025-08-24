from random import randrange, seed

kind = {"heart", "diamond", "spade", "club"}
number = {"ace", "jack", "king", "queen", 2, 3, 4, 5, 6, 7, 8, 9, 10}

deck = {(k,n) for k in kind
        for n in number}

player1 = set()
player2 = set()
list_deck = list(deck)

for i in range(5):
    pos1 = randrange(len(list_deck))
    player1.add(list_deck.pop(pos1))
    pos2 = randrange(len(list_deck))
    player2.add(list_deck.pop(pos2))

print(player1)
print()
print(player2)
print()

# kare assou

cnt = 0
for card in player1:
    if card[1] == "ace":
        cnt += 1
if cnt == 4:
    print("Player 1 has kare!")

cnt = 0
for card in player2:
    if card[1] == "ace":
        cnt += 1
if cnt == 4:
    print("Player 2 has kare!")

# kenta
# player 1
hand_numbers = []
for card in player1:
    if card[1] == "ace":
        hand_numbers.append(1)
    elif card[1] == "jack":
        hand_numbers.append(11)
    elif card[1] == "king":
        hand_numbers.append(12)
    elif card[1] == "queen":
        hand_numbers.append(13)
    else:
        hand_numbers.append(card[1])

hand_numbers.sort()
print(hand_numbers)

for pos in range(4):
    if hand_numbers[pos] != hand_numbers[pos+1] - 1:
        break
    else:
        print("Player 1 has kenta!")

# player 2
hand_numbers = []
for card in player2:
    if card[1] == "ace":
        hand_numbers.append(1)
    elif card[1] == "jack":
        hand_numbers.append(11)
    elif card[1] == "king":
        hand_numbers.append(12)
    elif card[1] == "queen":
        hand_numbers.append(13)
    else:
        hand_numbers.append(card[1])

hand_numbers.sort()
print(hand_numbers)

for pos in range(4):
    if hand_numbers[pos] != hand_numbers[pos+1] - 1:
        break
    else:
        print("Player 2 has kenta!")