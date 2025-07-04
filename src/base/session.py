import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

from .base import Base  # noqa

from .tables import Scaffold  # noqa

from dotenv import load_dotenv

load_dotenv(override=True)

# Create engine and Base
current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

engine = create_engine(
    f"sqlite:///{parent_dir}/db/cyber.db",
    connect_args={"check_same_thread": False},
)


def initialize_session():
    """
    Returns a new thread-safe session.
    """

    # Session factory
    SessionFactory = sessionmaker(bind=engine)

    # Create tables
    Base.metadata.create_all(engine)

    assert len(Base.metadata.tables.keys()) > 0

    session = SessionFactory()

    try:
        yield session
    except:
        session.rollback()
    finally:
        session.commit()
        session.close()
