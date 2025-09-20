from random import randrange

def quick_short(array, start, finish):

    def partition(array, start, finish):

        pivot = array[start]
        less = [elem for elem in array[start+1:finish+1] if elem < pivot]
        greaterequal = [elem for elem in array[start+1:finish+1] if elem >= pivot]

        new_array = less + [pivot] + greaterequal

        for i in range(start, finish+1):
            array[i] = new_array.pop(0)

        return start + len(less)

    def quick_sort_rec(array, start=0, finish=len(array)-1):
      
        if start < finish:
        
            pos = partition(array, start, finish)

            quick_sort_rec(array, start, pos-1)
            quick_sort_rec(array, pos+1, finish)

    quick_short(array, 0,len(array)-1)

array = [randrange(100) for _ in range(20)]

print(array)

new_array = quick_short(array, 0,len(array)-1)
print(f"{new_array.strip('[')}")