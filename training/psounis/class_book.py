class Author:
    def __init__(self, full_name, email):
        self.full_name = full_name
        self.email = email

class Book:
    def __init__(self, title, authors, price : float, copies : float):
        self.authors = authors
        self.price = price
        self.title = title
        self.copies = copies

    def print_full_title(self):
        authors_name = [author.full_name for author in self.authors]
        print(f"'{self.title}' by ", end="")
        print(", ".join(authors_name))

    def add_author(self, author):
        self.authors.append(author)

a1 = Author("Irgy Slj", "irg@gmail.com")
a2 = Author("Mikle Miky", "mik@gmail.com")
a3 = Author("Theo S", "theo@gmail.com")

b1 = Book("The Ideas Book", [a1, a2], 10.34, 1000)
b1.add_author(a3)

print(b1.print_full_title())