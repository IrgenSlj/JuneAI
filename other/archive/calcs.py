'''
seconds = 785627

hours = seconds // 3600
minutes = seconds % 3600

#72000 |_ 3600
#    0     2

print(f"{seconds} seconds is {hours} hours and {minutes} minutes")

x = 12356
y = x * 0.01
d = x + y
print(f"Arxiko poso {x} Euro")
print(f"Etisia auksisi {y} Euro")
print(f"Sunoliko poso meta apo ena xrono {d} Euro")
'''

hours = 10
minutes = 50
seconds = 15

total_seconds = ((hours * 60) * 60) + (minutes * 60) + seconds

print(f"Total seconds is {total_seconds}")