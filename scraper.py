from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os
import cv2
import sys
import shutil
import time
import requests
import random
from urllib import parse


agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:65.0) Gecko/20100101 Firefox/65.0"

headers = {
	'User-Agent': agent
}
def extract_url(str):
	str = str[str.index("imgurl") + 7:]
	str = str[:str.index("&imgrefurl=")]
	url = parse.unquote(str)
	return url

print("Please enter a search term")
search_term = input("> ")

stripped_search = search_term.strip()
print("Enter directory to place it or leave it blank to take term name")
directory = input("> ")
stripped_directory = directory.strip()

if len(stripped_search) == 0 or stripped_search == "":
	print("Search term is empty, exiting...")
	sys.exit()

url_search_term = search_term.replace(" ", "+")
no_space_search_term = search_term.replace(" ", "")
if len(stripped_directory) == 0 or stripped_directory == "":
    directory = search_term.lower().replace(" ", "_")
else:
    directory = stripped_directory
image_directory = "training_data/" + directory
if not os.path.exists(image_directory):
    os.makedirs(image_directory)

	


url = "https://images.google.com/"

profile = webdriver.FirefoxProfile()
profile.set_preference("general.useragent.override", agent)
driver = webdriver.Firefox(profile, executable_path=r'C:\Program Files (x86)\Gecko\geckodriver.exe')
driver.get(url)

search_input = None

time.sleep(1)

for input in driver.find_elements_by_tag_name('input'):
	if input is not None:
		title = input.get_attribute('title')
		if title is not None:
			if title == "Search":
				search_input = input
				break
				

search_input.send_keys(search_term)
search_input.send_keys(Keys.ENTER)

def scroll_page():
	for i in range(0,25):
		driver.execute_script("window.scrollBy(0, 1000);")
		time.sleep(1)

def show_more_results():
	inputs = driver.find_elements_by_tag_name('input')
	
	for input in inputs:
		if input is not None:
			value = input.get_attribute('value')
			if value is not None:
				value = value.lower()
				if "show more" in value:
					return input
	return None

scroll_page()

show_more = show_more_results()

if show_more is not None:
	show_more.click()
	scroll_page()
	

	
elements = driver.find_elements_by_tag_name('a')


index = 0
id = random.randint(1,101)
for element in elements:
	if element is not None:
		href = element.get_attribute('href')
		if href is not None:
			if not href.startswith("https://www.google.com/imgres?imgurl="):
				continue
			
			print(href)
			img = element.find_elements_by_tag_name('img')
			if len(img) == 0:
				continue
			img = img[0]
			href = extract_url(href)
			has_term = no_space_search_term in href
			
			if img is not None and not has_term:
				alt = img.get_attribute('alt')
				if alt is not None:
					alt = alt.lower()
					print(alt)
					if search_term.lower() in alt:
						has_term = True
			
			if not has_term:
				continue
			
			response = None
			try:
				response = requests.get(href, headers=headers)
			except Exception as ex:
				print(ex)
						
			if response is None:
				continue
			if response.status_code == 200:
				file_name = str(id) +stripped_search + str(index) + ".jpg"
				with open(image_directory + "/" + file_name, "wb") as f:
					f.write(response.content)
					f.close()
				index += 1
				img = cv2.imread(image_directory + "/" + file_name)
				horizontal_img = cv2.flip( img, 1 )
				cv2.imwrite(image_directory + "/" + "flipped" + file_name, horizontal_img)