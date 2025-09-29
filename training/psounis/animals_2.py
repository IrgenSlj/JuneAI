class Animal:
    def __init__(self):
        pass

    def make_sound(self):
        return f" "
    
class Cat(Animal):
    def __init__(self):
        pass
    
    def make_sound(self):
        return "Meow"
    
class Himal_Cat(Cat):
    def __init__(self):
        pass

    def make_sound(self):
        return super().make_sound() + " " + "Miouw Miouw"
    
class Dog(Animal):
    def __init__(Self):
        pass

    def make_sound(self):
        return "Woof woof"
    
class Doberman(Dog):
    def __init__(self):
        pass
    
class KingDoberman(Doberman):
    def __init__(self):
        pass

    def make_sound(self):
        return super().make_sound() + " " + "WOOOOAAAAAFFF"
    

cat = Cat()
him = Himal_Cat()
dog = Dog()
dob = Doberman()
king = KingDoberman()

print(cat.make_sound())
print(him.make_sound())
print(dog.make_sound())
print(dob.make_sound())
print(king.make_sound())