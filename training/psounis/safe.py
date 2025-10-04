def safe_divide():
    try:
        nom = int(input("Give nominator: "))
        denom = int(input("Give denominator: "))
        res = nom / denom
    except ZeroDivisionError:
        print("Denom can't be 0")
    except Exception as e:
        print(f"Bro there was {str(e)}")
    else:
        print(res)
    finally:
        pass

safe_divide()