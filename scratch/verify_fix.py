import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from database.db import get_db, create_session, get_all_categories

def test_categories():
    db = get_db()
    
    # Check current categories
    initial_cats = get_all_categories()
    print(f"Initial categories: {initial_cats}")
    
    # Create a new session with a unique category
    unique_cat = "Quantum Computing Expert"
    print(f"Creating session with category: {unique_cat}")
    
    session_id = create_session(
        job_description="Need a quantum computing expert.",
        job_category=unique_cat.lower(), # The app stores it lowercase
        num_resumes=0
    )
    
    # Check categories again
    updated_cats = get_all_categories()
    print(f"Updated categories: {updated_cats}")
    
    if unique_cat.lower() in updated_cats:
        print("SUCCESS: New category found in database list!")
    else:
        print("FAILURE: New category NOT found.")
        
    # Clean up (optional but good)
    db.sessions.delete_one({'_id': session_id}) if hasattr(db.sessions, 'delete_one') else None

if __name__ == "__main__":
    test_categories()
