merch = {
    "book": 10.10,
    "parsley": 0.22,
    "cement": 5.17,
    "cd": 0.05
}

while True:
    rate = float(input(
        "Give client rating (0 - 5)\nor type 10 to exit: \n"))
    while rate <= 0 or 10 > rate > 5 or rate > 10:
        rate = float(input(
            "Give client rating only between 0 and 5\nor type 10 to exit: \n"))
        
    if rate == 10:
        break
    else:
        new_value = {key:(value * rate) for key, value in merch.items()}
        print()
        for key in new_value:
            print(f"{key} = {round(new_value[key], 2)} Euro")
        print()