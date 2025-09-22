class Time:
    def __init__(self, hour, minute, second):
        self.hour = hour
        self.minute = minute
        self.second = second

    def __str__(self):
        return f"{str(self.hour).zfill(2)}:" \
            f"{str(self.minute).zfill(2)}:" \
            f"{str(self.second).zfill(2)}"
    
    def __repr__(self):
        return f"Time {self.hour}, {self.minute}, {self.second}"

class Date:
    def __init__(self, day, month, year):
        self.day = day
        self.month = month
        self.year = year

    def __str__(self):
        return f"{self.day}/{self.month}/{self.year}"
    
    def __repr__(self):
        return f"Date({self.day}, {self.month}, {self.year})"
    
    def __eq__(self, other):
        return self.day == other.day and self.month == other.month and self.year == other.year
    
class DateTime:
    def __init__(self, date, time):
        self.date = date
        self.time = time

    def __str__(self):
        return f"{self.date}, {self.time}"
    
    def __repr__(self):
        return f"Date {self.date}, Time {self.time}"

d = Date(2, 7, 1989)
t = Time(5, 55, 00)
date = DateTime(d, t)

print(repr(t))