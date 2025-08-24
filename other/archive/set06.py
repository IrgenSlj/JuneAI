from random import randrange
N = 30
students = set()
math1 = set()
math2 = set()
geo1 = set()
geo2 = set()

for n in range(1, N+1):
    students.add(f"student{n}")

students_list = list(students)


"""
repeat N/2 times
pick a random position, pop this pupil
make a team
add this team to teams
"""

def rand_half(students_list):
    math1 = set()
    ath2 = set()
    for _ in range(N // 2):
        pos1 = int(randrange(0,len(students_list)))
        pos2 = int(randrange(0,len(students_list)))
        st1 = students_list.pop(pos1)
        st2 = students_list.pop(pos2)
        math1.add(st1)
        math2.add(st2)
        return math1, math2
    
math1, math2 = rand_half(students_list)

print(math1)
print(math2)

