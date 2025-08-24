class C:
    cnt = 0

    def __init__(self):
        C.cnt += 1

    def __del__(self):
        C.cnt -= 1

o1 = C()
o2 = C()
o3 = C()

print(C.cnt, o1.cnt, o2.cnt)
del o3

print(C.cnt, o1.cnt, o2.cnt)

