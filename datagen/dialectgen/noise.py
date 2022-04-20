import random
from dataset_readers.dataset_util.spider_utils import is_number

def duplicate_one_word(dialect: str, exclude: set):
    dialect_list = dialect.split()
    length = len(dialect_list)   
    duplicate_word = dialect_list[random.randint(0, length - 1)]
    while True:
        if duplicate_word not in exclude and not is_number(duplicate_word):
            break
        else:
            duplicate_word = dialect_list[random.randint(0, length - 1)]
    insert_position = random.randint(0, length)
    dialect_list.insert(insert_position, duplicate_word)
    return ' '.join(dialect_list)

def duplicate_multi_word(dialect: str, exclude: set, duplicate_num: int):
    for i in range(duplicate_num):
        dialect = duplicate_one_word(dialect, exclude)
    return dialect

def duplicate(dialect:str, exclude:set, percentage:float):
    dialect_list = dialect.split()
    length = len(dialect_list)
    if int(length * percentage) < 1:
        random_max = 2
    else:
        random_max = int(length * percentage)
    duplicate_num = random.randint(0, random_max)
    dialect = duplicate_multi_word(dialect, exclude, duplicate_num)
    return dialect


# dialect = 'select name of singer whose age is 30 and hobby is running'
# exclude = {'of', 'is', 'and', 'whose', '30', 'select'}
# result = duplicate(dialect, exclude, percentage=0.2)
# print(result)

