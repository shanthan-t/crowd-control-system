"""
auth.py — Sentinel Authentication Module
Unified user model: { username, password (bcrypt), role: "admin"|"user" }
"""

import pymongo
import bcrypt
import os

# --- Configuration ---
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "sentinel_auth"
COLLECTION_NAME = "users"

# Default admin credentials (seeded on startup if no admin exists)
DEFAULT_ADMIN_USER = os.environ.get("SENTINEL_ADMIN_USER", "admin")
DEFAULT_ADMIN_PASS = os.environ.get("SENTINEL_ADMIN_PASS", "admin123")


def get_collection():
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME]


# ── Seed ───────────────────────────────────────────────────────────────────

def seed_admin():
    """Migrate old schema docs and ensure an admin exists.
    Called once at FastAPI startup."""
    users = get_collection()

    # ── Migrate old-schema docs (is_admin flag → role field) ──────
    # Old schema: {username, password, is_admin: True/False}
    # New schema: {username, password, role: "admin"|"user"}
    old_admins = users.find({"is_admin": True, "role": {"$exists": False}})
    for doc in old_admins:
        users.update_one({"_id": doc["_id"]}, {"$set": {"role": "admin"}, "$unset": {"is_admin": ""}})
        print(f"[Auth] Migrated old admin: {doc['username']}")

    old_users = users.find({"role": {"$exists": False}})
    for doc in old_users:
        users.update_one({"_id": doc["_id"]}, {"$set": {"role": "user"}, "$unset": {"is_admin": ""}})
        print(f"[Auth] Migrated old user: {doc['username']}")

    # Remove duplicate usernames (keep first)
    pipeline = [{"$group": {"_id": "$username", "count": {"$sum": 1}, "ids": {"$push": "$_id"}}},
                {"$match": {"count": {"$gt": 1}}}]
    for dup in users.aggregate(pipeline):
        # Keep the first, delete the rest
        to_delete = dup["ids"][1:]
        users.delete_many({"_id": {"$in": to_delete}})
        print(f"[Auth] Removed {len(to_delete)} duplicate(s) for: {dup['_id']}")

    # ── Seed default admin if none exists ─────────────────────────
    existing = users.find_one({"role": "admin"})
    if existing:
        print(f"[Auth] Admin ready: {existing['username']}")
        return
    hashed = bcrypt.hashpw(DEFAULT_ADMIN_PASS.encode("utf-8"), bcrypt.gensalt())
    users.insert_one({
        "username": DEFAULT_ADMIN_USER,
        "password": hashed,
        "role": "admin",
    })
    print(f"[Auth] Seeded default admin: {DEFAULT_ADMIN_USER}")


# ── Registration (public users only) ──────────────────────────────────────

def register_user(username: str, password: str) -> tuple:
    """Register a new user with role='user'.

    Returns:
        (True, "") on success
        (False, error_message) on failure
    """
    users = get_collection()
    if users.find_one({"username": username}):
        return False, "Username already taken."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    users.insert_one({
        "username": username,
        "password": hashed,
        "role": "user",
    })
    return True, ""


# ── Verification ──────────────────────────────────────────────────────────

def verify_user(username: str, password: str) -> bool:
    """Returns True if credentials are valid."""
    users = get_collection()
    user = users.find_one({"username": username})
    if not user:
        return False
    return bcrypt.checkpw(password.encode("utf-8"), user["password"])


def get_user_role(username: str) -> str | None:
    """Returns 'admin', 'user', or None if not found."""
    users = get_collection()
    user = users.find_one({"username": username})
    if not user:
        return None
    return user.get("role", "user")


# ── Legacy compatibility (used by control_room) ──────────────────────────

def is_admin(username: str) -> bool:
    return get_user_role(username) == "admin"


def create_user(username, password):
    """Legacy — creates user with no role (defaults to 'user')."""
    ok, _ = register_user(username, password)
    return ok
