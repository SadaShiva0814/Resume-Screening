"""
Export / Import utility for the JSON document store.
Can be used to backup data or migrate between environments.
"""
import json
import os

STORE_PATH = os.path.join(os.path.dirname(__file__), 'database', 'store.json')
SEED_PATH = os.path.join(os.path.dirname(__file__), 'database', 'seed_data.json')


def export_store(output_path=None):
    """Export the current store.json to a readable backup."""
    output_path = output_path or 'database/backup_data.json'
    if not os.path.exists(STORE_PATH):
        print(f"No store.json found at {STORE_PATH}")
        return

    with open(STORE_PATH, 'r') as f:
        data = json.load(f)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    total = sum(len(v) for v in data.values() if isinstance(v, list))
    print(f"Exported {total} documents to {output_path}")


def import_seed_to_store():
    """Import seed_data.json into store.json (useful for first-time setup)."""
    if not os.path.exists(SEED_PATH):
        print(f"No seed_data.json found at {SEED_PATH}")
        return

    if os.path.exists(STORE_PATH):
        print(f"store.json already exists. Delete it first if you want to re-seed.")
        return

    with open(SEED_PATH, 'r') as f:
        data = json.load(f)

    total = sum(len(v) for v in data.values() if isinstance(v, list))
    print(f"Loaded {total} documents from seed_data.json")

    # The app will auto-convert seed_data.json to store.json on first run.
    # But we can do it manually here too.
    from database.db import get_db
    db = get_db()
    print(f"Store initialized. Data will be persisted to {STORE_PATH}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'import':
        import_seed_to_store()
    else:
        export_store()
