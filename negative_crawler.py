'''
author : 백승윤
negative class에 해당하는 train/val 데이터셋을 수집하는 파이썬 파일입니다.
오타와 emoji등이 많고, 지급된 dataset의 negative class와 유사한 특징을 보이는
메이플스토리M 네이버카페 게시판의 데이터를 수집했습니다.

페이지를 탐색하는 시간을 랜덤하게 주고, 스크롤링을 하고,
랜덤하게 창을 껐다 키는 등의 기법으로 block을 피했습니다.
'''
import sys
import pandas as pd
import numpy as np
import random, math

from selenium import webdriver 

import time
from tqdm import tqdm
import crawling_utils as cu # custom utils

# define chrome webdriver
def create_chrome(win_w, win_h):
    
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--window-size={}x{}'.format(win_w, win_h))
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument("--lang=ko_KR")

    chrome_options.add_argument("--incognito") # secret mode

    # 웹드라이버 실행
    path = 'C:/Users/sybaek/Downloads/chromedriver_win32/chromedriver.exe' # chrome driver 위치
    driver = webdriver.Chrome(path, chrome_options=chrome_options)
    time.sleep(6)
    driver.set_window_size(win_w, win_h)

    driver.execute_script("Object.defineProperty(navigator, 'plugins', {get: function() {return[1, 2, 3, 4, 5];},});")
    return driver

# driver 선언
driver = create_chrome(900, 900)

# config
csv_name = 'negative30k' + '.csv'
get_data_num = 10000 # 가져올 데이터 갯수 : 목표 갯수에 도달하면 프로그램을 종료합니다.
# 일반적으로 네이버 카페 게시판의 경우 1000페이지를 초과할 수 없기 때문에 10000개정도 선에서 모아줍니다.
# test_mode = True


# crawling url : maplestory m naver fan cafe
url = 'https://cafe.naver.com/nexonmaplestorym?iframe_url=/ArticleList.nhn%3Fsearch.clubid=28385054%26search.boardtype=L%26search.totalCount=151%26search.page='


# 지나친 어뷰징 방지를 위해 sleep time 설정 (랜덤하게)
sleep_time = [round(random.random(),4) * 2 + 1 for i in range(int(1000))]
scroll_range = [round(random.random(),2) * 800 + 80 for i in range(int(1000))]

def main(driver, url, sleep_time) :
    start = time.time() # for checking crawling time...
    rec_frame = pd.DataFrame(columns = ['txt', 'label'])
    rec_list = []
    page_num = 1
    for sleep, scroll in zip(tqdm(sleep_time), scroll_range) :
        # iter pg_num
        page_num += 1 # 공지가 없는 2pg 부터 색인
        driver.get(url + str(page_num))
        # wait!
        time.sleep(0.5)
        # scroll per page 2 ~ 5 time randomly
        for random_scroll in range(random.randint(0,5)) : 
            time.sleep(sleep * random.random())
            driver.execute_script("window.scrollTo(0, {})".format(scroll* random.random()))
        # scrape the whole data
        scraped_data = cu.get_data(driver)
        if scraped_data is None :
            print('Got none data!')
            pass
        else : 
            '''
            scraped data split and save (drop duplicates)
            '''
            [rec_list.append(cu.split_data(txt)) for txt in scraped_data if cu.split_data(txt) not in rec_list]

        end = time.time()

        # incase searching is banned, exit the program and print the result
        if driver.current_url.startswith('https://cafe.naver.com/nexonmaplestorym?iframe_url=/ArticleList.nhn%3Fsearch') == False :
            print('Scraping is blocked!')
            print('time elapsed : {:.3f} min, num of data : {}'.format((end - start) / 60, len(rec_list)))
            # record and break
            rec_frame['txt'] = rec_list
            rec_frame['label'] = [0 for i in range(len(rec_list))]
            rec_frame.to_csv(csv_name)
            break

        elif len(rec_list) >= get_data_num : # 정해진 갯수만큼 모으면 저장하고 중단
            # record and break
            print('time elapsed : {:.3f} min, num of data : {}'.format((end - start) / 60, len(rec_list)))
            rec_frame['txt'] = rec_list
            rec_frame['label'] = [0 for i in range(len(rec_list))]
            rec_frame.to_csv(csv_name)
            break
        # display process and save frequently
        if (page_num - 1) % 20 == 0 :
            print('time elapsed : {:.3f} min, num of data : {}'.format((end - start) / 60, len(rec_list)))
            rec_frame['txt'] = rec_list
            rec_frame['label'] = [0 for i in range(len(rec_list))]
            rec_frame.to_csv(csv_name)
            rec_frame = pd.DataFrame(columns = ['txt', 'label'])

        # reopen the window
        if page_num % random.randint(30,50) == 0 :
            # close the window
            driver.close()
            # wait 15 ~ 25 sec
            time.sleep(random.randint(9,15))
            # reopen the window
            driver = create_chrome(900, 900)
            time.sleep(6)
            # back to url
            driver.get(url)

# 실행
main(driver, url, sleep_time)