import requests

request = requests.get('http://api.open-notify.org')
print(request.text)
print(request.status_code)
people = requests.get('http://api.open-notify.org/astros.json')
print(people.text)
