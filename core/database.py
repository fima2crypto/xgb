from sqlalchemy import create_engine
import psycopg2

# ======================================================
# DB CONFIG
# ======================================================
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "katalott",
    "user": "postgres",
    "password": "bin",
}

# ======================================================
# PSYCOPG2 CONNECTION
# ======================================================
def get_db():
    try:
        conn = psycopg2.connect(
            **DB_CONFIG,
            connect_timeout=3
        )
        return conn

    except Exception as e:
        print(f"❌ LỖI KẾT NỐI DB: {e}")
        raise e

# ======================================================
# SQLALCHEMY ENGINE
# ======================================================
def get_engine():

    url = (
        f"postgresql+psycopg2://"
        f"{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}"
        f"/{DB_CONFIG['dbname']}"
    )

    return create_engine(url)