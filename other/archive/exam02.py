bests = ["John", "Mike", "Pam"]
inv = ["John", "Mike", "Pam", "Jonny", "Mikky", "Pamela", "Johny", "Mikey", "Pamy"]

cnt = 0

for friend in inv:
    if friend in bests:
        cnt += 1
        
'''

for i in range(len(inv)):
    for j in range(len(bests)):
        if inv[i] == bests[j]:
            cnt += 1
'''
if cnt <= 2:
    print("Party invitation rejected")
else:
    print("Party invitation accepted")