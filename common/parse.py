import os
from argparse import ArgumentTypeError

def yes_or_no(question):
    # could this overflow the stack if the user was very persistent?
    reply = str(input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return yes_or_no("Please enter ")

def numeric_min_checker(minimum, message, numeric_type=int):
    def check_number(n):
        n = numeric_type(n)
        if n < minimum:
            raise ArgumentTypeError("{msg}: got {got}, minimum is {minimum}".format(
                msg=message, got=n, minimum=minimum
            ))
        return n
    return check_number

def path_exists_checker(check_dir=True, make_absolute=True, make_if_empty=False):
    def _func(path):
        path = os.path.expanduser(path)
        if os.path.exists(path):
            if check_dir:
                if not os.path.isdir(path):
                    raise ArgumentTypeError("path {pth} exists, but isn't a directory".format(pth=path))
            elif not os.path.isfile(path=path):
                raise ArgumentTypeError("path {pth} exists, but isn't a file".format(pth=path))
        elif check_dir and make_if_empty:
            os.makedirs(name=path)
        else:
            raise ArgumentTypeError("path {pth} doesn't exist on filesystem".format(pth=path))
        if make_absolute:
            path = os.path.abspath(path=path)
        return path
    return _func

def non_empty_string_checker(string):
    if len(string) == 0:
        raise ArgumentTypeError("string is empty!")
    return string
