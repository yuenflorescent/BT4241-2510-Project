from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd

from bs4 import BeautifulSoup
import time

# 1. Setup Selenium
driver = webdriver.Chrome()

txt_output = ""
try:
    years = list(range(2000,2014))
    for year in years:
        url = f"https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate={year}"
        driver.get(url)
        
        wait = WebDriverWait(driver, 10)
        
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#tb_list > div.tb_list.type02.d_song_list")))
        
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        
        songs = []
        for info in soup.select("[class~=lst50]"):
            artist_tag = info.css.select_one("td:nth-child(4) > div > div > div:nth-child(3) > div.ellipsis.rank02 > a").get_text()
            title_tag_container = info.css.select_one("td:nth-child(4) > div > div > div.ellipsis.rank01 > span > strong > a")
            if title_tag_container:
                title_tag = info.css.select_one("td:nth-child(4) > div > div > div.ellipsis.rank01 > span > strong > a").get("title")
            else:
                title_tag = info.css.select_one("td:nth-child(4) > div > div > div.ellipsis.rank01 > span > span > div").get_text()
            songs.append(artist_tag + " " + title_tag)
        

        toggle_button = driver.find_element(By.CSS_SELECTOR, "#tb_list > div.paginate.chart_page > span > a")
        toggle_button.click()
        
        # Wait for new elements
        time.sleep(2)  # or better: wait for an element change / new elements
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#tb_list > div.tb_list.type02.d_song_list")))
        
        # Parse again
        html2 = driver.page_source
        soup2 = BeautifulSoup(html2, "html.parser")
        for info in soup2.select("[class~=lst100]"):
            artist_tag = info.css.select_one("td:nth-child(4) > div > div > div:nth-child(3) > div.ellipsis.rank02 > a").get_text()
            title_tag_container = info.css.select_one("td:nth-child(4) > div > div > div.ellipsis.rank01 > span > strong > a")
            if title_tag_container:
                title_tag = info.css.select_one("td:nth-child(4) > div > div > div.ellipsis.rank01 > span > strong > a").get("title")
            else:
                title_tag = info.css.select_one("td:nth-child(4) > div > div > div.ellipsis.rank01 > span > span > div").get_text()
            songs.append(artist_tag + " " + title_tag)
        
        print(f"Total songs scraped for {year}", len(songs))
        txt_output += str(year)
        txt_output += "\n"
        txt_output += str.join("\n", songs)
        txt_output += "\n"
        txt_output += "\n"
    
    with open("korean_charts_artists_and_songs.txt", "w") as text_file:
        text_file.write(txt_output)
    print("done")

finally:
    driver.quit()