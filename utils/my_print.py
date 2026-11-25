# -*- coding: utf-8 -*-

# color_list = {
#         "black": 0,
#         "red": 1,
#         "green": 2,
#         "yellow": 3,
#         "blue": 4,
#         "purple": 5,
#         "cyan": 6,
#         "gray": 7,
#         "white": 8
#     }


def my_print(head, head_type='info', content=None):
    if content is None:
        content = ''
    if head_type == 'info':
        bg = 42
        word = 38
    elif head_type == 'warn':
        bg = 43
        word = 31
    elif head_type == 'err':
        bg = 41
        word = 38
    elif head_type == 'data':
        bg = 47
        word = 30
    else:
        bg = 45
        word = 38
    print(f"\n\033[1;{bg}m\033[1;{word}m   {head}   \033[0m\033[0m\n{content}\n")
