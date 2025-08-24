class Philosopher:
    def __init__(self, full_name, epoch, books, ideology, noise):
        self.name = full_name
        self.epoch = epoch
        self.books = books
        #self.books.extend(books)
        self.ideology = ideology
        self.doubts = noise
        #self.doubts.extend(noise)


plato = Philosopher("Plato", "Ancient Greek", 
                    ["Republic", "Phaedon", "Memoires"], "Platonism", ["Socrates", "others"])

print(plato.books)
print(plato.name)
doubt_list = []
doubt_list.extend(element for element in plato.doubts)
print(doubt_list)