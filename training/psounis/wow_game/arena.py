from character import Character
from random import randrange

class Arena:

    def __init__(self, team_a, team_b):
        self.team_a = team_a
        self.team_b = team_b

    def print_state(self):
        print("TEAM A")
        for character in self.team_a:
            character.print()
        print("-" * 10)

        print("TEAM B")
        for character in self.team_b:
            character.print()

    def play(self):
        time = -1
        while True:
            time += 1
            print("=" * 10)
            print("Time = " + str(time))
            print("=" * 10)
            self.print_state()

            # create list of characters to play
            chars_to_play = []

            for character in self.team_a:
                if character.delay == 0:
                    chars_to_play.append(("A", character))

            for character in self.team_b:
                if character.delay == 0:
                    chars_to_play.append(("B", character))

            # active characters attack

            for character in chars_to_play:
                attacking = character[1]
                if attacking is "A":
                    defending = randrange(len(self.team_a))
                else:
                    defending = randrange(len(self.team_b))

                damage = attacking.attack()
                defending.health -= damage
                print(f"{attacking.character_name} dealt {damage} to {defending.character_name}")

            # check for dead characters

            for pos in range(len(self.team_a) -1, -1, -1):
                if self.team_a[pos].is_dead():
                    self.team_a.pop(pos)
                    print(f"{self.team_a[pos]} is dead!")

            
            for pos in range(len(self.team_b) -1, -1, -1):
                if self.team_b[pos].is_dead():
                    self.team_b.pop(pos)
                    print(f"{self.team_b[pos]} is dead!")

            if len(self.team_a) == 0:
                print(f"Team A won!!")
                break
            elif len(self.team_b) == 0:
                print("Team B won")
                break
            else:
                pass

            # end round
            for character in self.team_a:
                character.end_round()

            for character in self.team_b:
                character.end_round()

            input("Press ENTER to continue")