
def get_speed(datas):
    """
From the array of characters read by UART, 
returns the speed of the wheels as a number, if possible.
    """
    return float(''.join(datas[:]))
