class Stack:

    def __init__(self):
        self.array = []

    def push(self, item):
        self.array.append(item)

    def pop(self):

        if not self.array:
            return None
        else:
            self.array.pop()

    def length(self):
        return len(self.array)

