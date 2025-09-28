from random import randrange
from equipment import Equipment

class Character:
    def __init__(self, character_name, equipment, attack_speed=2, delay=0):
        self.character_name = character_name
        self.equipment = equipment
        self.max_health = 100 * self.equipment.cape 
        self.health = 100 * self.equipment.cape
        self.attack_speed = attack_speed
        self.delay = delay

    def attack(self):
        self.delay = 5 - self.attack_speed
        return randrange(3, 11) * self.equipment.sword
    
    def is_dead(self):
        return self.health <= 0
    
    def end_round(self):
        self.healh = self.health +1 if self.health + 1 <=100 else 100
        self.delay -= 1

    def __str__(self):
        return f"{self.character_name} H: {self.health} D: {self.decay}"
    
    def __repr__(self):
        return f"Character {self.character_name}, {self.attack_speed}, {self.delay}, {self.health}"
    def __iadd__(self, other):
        self.health += other
        return self

    def __isub__(self, other):
        self.health == other
        return self 