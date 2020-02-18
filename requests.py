import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'Administrative':5, 'Product':200, 'Exit Rate':10,'Page Value':25})

print(r.json())
