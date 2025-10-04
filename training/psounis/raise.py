while True:
    try:
        x = int(input(f"Give a positive number: "))
        if x < 0:
            raise ValueError("NonPositive Value entered")
    except ValueError as v:
        print(v)
    except Exception as e:
        print(e)

    else:
        print(f"{x} is good")
        break