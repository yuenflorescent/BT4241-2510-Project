from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

#mw-content-text > div > table > tbody > tr:nth-child(2) > td:nth-child(2)
#mw-content-text > div > table > tbody > tr:nth-child(3) > td:nth-child(3)

driver = webdriver.Chrome()
txt_output = " "

try:
    years = list(range(2000,2014))
    for year in years:
        url = f"https://987fmdotsg.fandom.com/wiki/987FM_Top_100_{year}"
        driver.get(url)
        
        wait = WebDriverWait(driver, 10)
        
        # Wait for the list items to load (adjust selector if needed)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "article-table")))
        
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        
        songs = []
        
        first_element = soup.find('table', class_='article-table')
        
        count = 0 # skip the header
        for info in first_element.select("tr"):
            if count == 0:
                count += 1
                continue
            artist_tag_container = info.css.select_one("td:nth-child(3)")
            if artist_tag_container:
                artist_tag = artist_tag_container.get_text()
            else:
                artist_tag = prev_artist_tag
            title_tag = info.css.select_one("td:nth-child(2)").get_text()
            songs.append(artist_tag.strip() + " " + title_tag.strip().replace('“', '').replace('”', ''))
            prev_artist_tag = artist_tag
            
        print(f"Total songs scraped for {year}:", len(songs))
        txt_output += str(year)
        txt_output += "\n"
        txt_output += str.join("\n", songs)
        txt_output += "\n"
        txt_output += "\n"
        
    with open("sin_charts_artists_and_songs.txt", 'w') as text_file:
        text_file.write(txt_output)

finally:
    driver.close()