
def get_speed(datas):
    """
From the array of characters read by UART, 
returns the speed of the wheels as a number, if possible.
    """
    if datas[0] == '-':
        value = - float(''.join(datas[1:]))
    else:
        value = float(''.join(datas[:]))
    return value
    

