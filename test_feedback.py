import urllib.request, json
from pymongo import MongoClient

db = MongoClient('mongodb://localhost:27017/').resume_screening
c = db.candidates.find_one()
session_id = str(c['session_id'])
candidate_id = str(c['_id'])

req = urllib.request.Request(
    'http://127.0.0.1:5001/api/feedback', 
    data=json.dumps({'session_id': session_id, 'candidate_id': candidate_id, 'action': 'shortlisted'}).encode('utf8'), 
    headers={'content-type': 'application/json'}
)
try:
    print(urllib.request.urlopen(req).read().decode('utf8'))
except urllib.error.HTTPError as e:
    print(e.read().decode('utf8')[:2000])
