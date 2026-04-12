import os
from pymongo import MongoClient
import sys

def test_connection():
    uri = "mongodb+srv://isigodin678_db_user:B9gLU8k4dJHgwG4x@resume-screening.letslxf.mongodb.net/?appName=Resume-Screening"
    print(f"Testing connection to MongoDB Atlas...")
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        print("✅ Success! Connection to MongoDB Atlas established.")
        
        db = client['resume_screening']
        print(f"Checking database: {db.name}")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    if test_connection():
        sys.exit(0)
    else:
        sys.exit(1)
