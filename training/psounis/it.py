class MyIterator:
    def __iter__(self):
        self.n = 0
        return self
    
    def __next__(self):
        self.n += 1000
        if self.n <= 1000000:
            return list(range(self.n-1000, self.n))
        else:
            raise StopIteration
    
for next_data in MyIterator():
    for data in next_data:
        print(data)