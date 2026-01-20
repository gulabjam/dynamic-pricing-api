"""
MongoDB connection management module.
Provides singleton client and context manager for database connections.
"""
import os
from pymongo import MongoClient
from dotenv import load_dotenv
from contextlib import contextmanager

load_dotenv()
# MongoDB Configuration
MONGO_URI = os.getenv('MONGO_URL')
DB_NAME = "Users"
COLLECTION = "Demand_data"

# Singleton MongoDB client
_mongo_client = None


def get_mongo_client():
    """Get or create a MongoDB client instance (singleton pattern)."""
    global _mongo_client
    if _mongo_client is None:
        print("🔹 Connecting to MongoDB...")
        if not MONGO_URI:
            raise ValueError("❌ MONGO_URL environment variable is not set!")
        _mongo_client = MongoClient(MONGO_URI)
        # Test the connection
        try:
            _mongo_client.admin.command('ping')
            print("✅ MongoDB connection successful")
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            _mongo_client = None
            raise
    return _mongo_client


@contextmanager
def get_db_connection():
    """Context manager for MongoDB database connection."""
    client = get_mongo_client()
    db = client[DB_NAME]
    print(f"📦 Using database: {DB_NAME}, collection: {COLLECTION}")
    try:
        yield db
    except Exception as e:
        print(f"❌ Database error: {e}")
        raise


def close_connection():
    """Close the MongoDB connection. Call this when shutting down the application."""
    global _mongo_client
    if _mongo_client is not None:
        _mongo_client.close()
        _mongo_client = None
        print("🔹 MongoDB connection closed")
