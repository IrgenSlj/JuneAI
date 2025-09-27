class Queue:
    def __init_(self):
        self.array = []

    def  enqueue(self, item):
        self.array.append(item)

    def pop(self):
        if not self.array:
            return None
        else:
            return self.array.pop(0)
        
    def __str__(self):
        return ", ".join(self.array)
    
    def __add__(self, other):
        new_q = Queue()
        new_q.array = self.array[:]
        new_q.enqueue(other)
        return new_q
    
    def __iadd__(self, other):
        self.enqueue(other)
        return self
    
    def __neg__(self):
        return self.pop()
    
    def __len__(self):
        return len(self.array)
        