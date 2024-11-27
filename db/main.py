import gspread, time, os

#https://docs.google.com/spreadsheets/d/19FnmST_Ea0lyzBS8Qg5Y3KjT5i1XQFaJaPpgdArDmI0/edit
key_path=os.path.join('tasks','view_tasks','db','Key.json')

secret_key = gspread.service_account(key_path) #подключение в json файлу библиотеки
sh = secret_key.open("Data") #открытие таблицы с таким-то названием
sh1=sh.get_worksheet(0) #определение рабочей страницы в таблице


def on_hold(seconds: int): time.sleep(seconds) #задержка для передачи излишних запросов серверу
    
def get_sh1() -> list[list]: #все значения стр. "Мероприятия"
    try: return sh1.get_values()[1:] #вывод общего списка мероприятий
    except gspread.exceptions.APIError:
        on_hold(10)
        return get_sh1()  

local_list=get_sh1() #вызов метода при запуске программы, сокращение количества обращений к базе данных

def check_(data: list, user: str) ->list[list]:
    list_=data #присваиваем список преподавателей локальному списку
    return [item for item in list_ if item[0] == user]

#print(check_(get_sh1(),'cup')) #пример вызова