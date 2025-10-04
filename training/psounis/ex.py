class MyException(Exception):
    def __init__(self, val, message):
        self.val = val
        self.message = message

    def __str__(self):
        return f"{self.message}: {str(self.val)} is not valid"
    
class ValueTooSmallError(Exception):
    def __init__(self, message):
        self.message = message
    
try:
    val = int(input("Give an integer: "))
    if val < 0:
        raise MyException(val, "Negative integers not valid")
    
except MyException as m:
    print(m.__str__())
else:
    print(val)