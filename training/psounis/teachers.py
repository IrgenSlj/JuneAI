import json
from teacher import Teacher

class Teachers:

    def __init__(self):
        
        try:
            with open("teachers.json") as f:
                teachers_list = json.load(f)

            self.teachers = []
            for teacher_dict in teachers_list:
                t = Teacher()
                t.from_dict(teacher_dict)
                self.teachers += [t]

        except FileNotFoundError:
            self.teachers = [] 

    def save_teachers_data(self):

        list_to_write = []
        for teacher in self.teachers:
            list_to_write += [teacher.to_dict()]

        with open("teachers.json", "w") as f:
            json.dump(list_to_write, f)

    def next_id(self):

        if not self.teachers:
            return 1001
        else:
            ids = []
            for t in self.teachers:
                ids.append(t.teacher_id)
            return max(ids) + 1
        
    def create_teacher(self, first_name, surname):

        for teacher in self.teachers:
            if teacher.teacher_name == first_name and teacher.teacher_surname == surname:
                print("Error Teacher already exists")
                return None
            
        t = Teacher(first_name, surname, next_id())
        self.teachers.append(t)
        return t
    
    def read_teacher(self, id):

        for teacher in self.teachers:
            if teacher.teacher_id == id:
                return teacher
            else:
                return None

    def update_teacher(self, teacher_id):
    
        for teacher in self.teachers:
            if teacher.teacher_id == teacher_id:
                t = teacher

                input_user = input("Would you like to update teacher's name or surname? ")
            
                if input_user == "name":
                    t.teacher_name = input("Enter new name: ")
                elif input_user == "surname":
                    t.teacher_name = input("Enter surname: ")

                break

    def delete_teacher(self, teacher_id):
        for i in range(len(self.teachers)-1):
            if self.teachers[i]["teacher_id"] == teacher_id:
                self.teachers.pop(i)
                return
        else:
            return None   