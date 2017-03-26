import argparse

def numeric_min_checker(minimum, message, numeric_type=int):
    def check_number(n):
        n = numeric_type(n)
        if n < minimum:
            raise argparse.ArgumentTypeError("{msg}: got {got}, minimum is {minimum}".format(
                msg=message, got=n, minimum=minimum
            ))
        return n
    return check_number
