class King:
    def __init__(self, name, kingdom):
        self.name = name
        self.kingdom = kingdom

    def rule(self):
        print("Now I rule")

class Philosopher:
    def __init__(self, school, work):
        self.school = school
        self.work = work
    
    def think(self):
        print("Now I think")

class Emperor(King, Philosopher):
    def __init__(self, name, kingdom, school, work):
        King.__init__(self, name, kingdom)
        Philosopher.__init__(self, school, work)

    def __str__(self):
        st = f"{self.name} of {self.kingdom}, \nrepresentative of {self.school}, \nwriter of {self.work}"
        return st


marcus = Emperor("Marcus Aurelius", "Roman Empire", "Stoicism", "Reflections")

print(marcus.__str__())
marcus.think()
marcus.rule()
marcus.think()