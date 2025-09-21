class Teacher:
    def __init__(self, teacher_name="",teacher_surname="", teacher_id=-1 ):
        self.teacher_name = teacher_name
        self.teacher_surname = teacher_surname
        self.teacher_id = teacher_id
        
    def from_dict(self, teacher_dict):
        self.teacher_name = teacher_dict["teacher_name"]
        self.teacher_surname = teacher_dict["surname"]
        self.teacher_id = teacher_dict["teacher_id"]

    def to_dict(self):
        teacher_dict = {"teacher_name":self.teacher_name,
                        "teacher_surname": self.teacher_surname,
                        "teacher_id": self.teacher_id}
        return teacher_dict
    
    def print_teacher(self):
        print(f"Name {self.teacher_name}")
        print(f"Surname {self.teacher_surname}")
        print(f"ID {self.teacher_id}")