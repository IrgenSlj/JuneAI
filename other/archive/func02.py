def input_float():
    while True:
        numb = input("Type a float number: ").strip()
        if "." in numb:
            parts = numb.split(".")
            if len(parts) > 2:
                print("Only one dot as in a float number please.")
            elif parts[0].isdigit() and parts[1].isdigit():
                return float(numb)
            else:
                print("Only one dot as in a float number please.")
        else:
            if numb.isdigit():
                return float(numb)
            else:
                print("Only digits please.")


x = input_float()
print(x)