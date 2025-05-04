# database/database.py
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as SessionClass # Use SessionClass alias
from contextlib import contextmanager

# Import settings
from config import settings

logger = logging.getLogger(__name__)

try:
    engine = create_engine(settings.DB_URL, echo=False, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("Database engine and session created successfully.")
except Exception as e:
    logger.error(f"Failed to create database engine or session: {e}")
    # Depending on the application, you might want to exit or handle this differently
    engine = None
    SessionLocal = None

@contextmanager
def get_db_session() -> SessionClass | None:
    """Provide a transactional scope around a series of operations."""
    if SessionLocal is None:
        logger.error("Database session not initialized.")
        yield None
        return

    db = SessionLocal()
    try:
        yield db
        db.commit() # Commit on successful exit
    except Exception as e:
        logger.error(f"Database session error: {e}", exc_info=True) # Log exception details
        db.rollback() # Rollback on error
        raise # Re-raise the exception for higher level handling if needed
    finally:
        db.close() # Always close the session

def get_engine_instance():
    """Return the created engine instance."""
    return engine

def create_database_tables():
    """Create database tables defined in models."""
    if engine is None:
        logger.error("Database engine not initialized. Cannot create tables.")
        return
    logger.info("Creating database tables if they don't exist...")
    from . import models # Import models here to avoid circular dependencies
    try:
        models.Base.metadata.create_all(bind=engine)
        logger.info("Database tables checked/created successfully.")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise