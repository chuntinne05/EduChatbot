import requests
import xml.etree.ElementTree as ET

SITEMAP_URL = "https://tuyensinh.hcmiu.edu.vn/post-sitemap.xml"

resp = requests.get(SITEMAP_URL)
root = ET.fromstring(resp.content)

urls = [elem.text.strip() for elem in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
print(f"Found {len(urls)} URLs")

with open("sitemap_links.txt", "w") as f:
    for url in urls:
        f.write(url + "\n")
print("Saved all URLs to sitemap_links.txt")
