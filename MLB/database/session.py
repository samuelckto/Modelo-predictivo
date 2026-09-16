from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker

from MLB.database.models import Base
from shared.paths import MLB_DATABASE_URL

_engine = None
_Session = None


def engine():
    global _engine, _Session
    if _engine is None:
        _engine = create_engine(MLB_DATABASE_URL, future=True,
                                connect_args={"check_same_thread": False, "timeout": 30})

        @event.listens_for(_engine, "connect")
        def _pragmas(dbapi, _):
            c = dbapi.cursor()
            c.execute("PRAGMA journal_mode=DELETE")   # portable entre carpetas montadas
            c.execute("PRAGMA foreign_keys=ON")
            c.close()

        _Session = sessionmaker(bind=_engine, future=True, expire_on_commit=False)
    return _engine


def init_db():
    Base.metadata.create_all(engine())
    return MLB_DATABASE_URL


@contextmanager
def session_scope():
    engine()
    s = _Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def table_names() -> list[str]:
    with engine().connect() as c:
        return sorted(r[0] for r in c.execute(
            text("select name from sqlite_master where type='table' order by name")))
