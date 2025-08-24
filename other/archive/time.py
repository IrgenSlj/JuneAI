class Time:
    def __init__(self, hour=0, min=0, sec=0):
        self.hour = self.__validate(hour, 0, 23)
        self.min = self.__validate(min, 0, 59)
        self.sec = self.__validate(sec, 0, 59)

    def set_hour(self, hour):
        self.hour = self.__validate(hour, 0, 23)

    def set_min(self, min):
        self.min = self.__validate(min, 0, 59)

    def set_sec(self, sec):
        self.sec = self.__validate(sec, 0, 59)

    def __validate(self, val, low, upp):
        if low <= val <= upp:
            return val
        else:
            return 0

    def total_seconds(self):
        return (self.hour * 3600) + (self.min * 60) + self.sec
    
    def print(self):
        print(f"The time is {self.hour} : {self.min} : {self.sec}")

    def next_second(self):
        tm = Time(self.hour, self.min, self.sec + 1)
        tm.print()
        return tm

t = Time(22, 55, 17)
t.print()
t.next_second()