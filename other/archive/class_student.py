from random import randrange
from datetime import datetime

class  Student:
    def __init_(self, name):
        self.name = name
        self.grade = -1
    
def grade_student(student):
    student.grade = randrange(0, 11)

def avarage(students):
    sum = 0
    for student in students:
        sum +=student.grade
    
    print(str(sum / len(students)))
    
names = [
    "Nick Bytes",
    "Nick Bytes",
    "Nick Bytes",
    "Nick Bytes",
    "Nick Bytes",
    "Nick Bytes",
    "Nick Bytes",
    "Nick Bytes",
    "Nick Bytes",
    "Nick Bytes"
    ]

students = [Student(names[i]) for i in range(len(names))]

for student in students:
    grade_student(student)

avarage(students)