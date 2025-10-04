def get_integer():
    while True:
        try:
            x = input("Give positive inty: ")
            if x is str:
                raise ValueError("string entered")
            elif not x.isdigit():
                raise ValueError("no digits entered")
            
            x = int(x)

        except ValueError as v:
            print("Wrong input, only positive digits please")
        except Exception as e:
            print(e)
        else:
            print(x)
            break

get_integer()