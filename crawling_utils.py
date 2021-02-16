import pandas as pd
import re
from selenium.webdriver.common.keys import Keys
import time
import random
# import matplotlib.pyplot as plt

# def input_keyword(driver, keywords, method) :
#     search_bar = driver.find_element_by_name('q')
#     search_bar.clear()
#     for kw in keywords :
#         search_bar.send_keys(kw)
#         time.sleep(0.035 * random.randint(1,15))
#     button = driver.find_element_by_css_selector('button.Tg7LZd')

#     if method == 0 :  # Enter
#         search_bar.send_keys(Keys.RETURN)
#     elif method == 1 : # Click
#         button.click()

def get_data(driver) :
    try :
        driver.switch_to_frame('cafe_main')
        return [elem.text for elem in driver.find_elements_by_css_selector('a.article')]
    except Exception as e :
        print(e)
        pass

def split_data(txt) :
    try :
        if '] ' in txt : # 게시판 내 설정 말머리 제거
            txt = txt.split('] ')[1]
        else :
            pass
    except Exception as e :
        print(e)
        pass
    return txt
    
