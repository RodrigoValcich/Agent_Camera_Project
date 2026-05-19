from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy import Column, String, Float, DateTime, Text, Integer, Boolean
from datetime import datetime


class Base(DeclarativeBase):
    pass


class Event(Base):
    __tablename__ = "events"

    id          = Column(String, primary_key=True)
    camera_id   = Column(String, nullable=False)
    camera_name = Column(String)
    event_type  = Column(String, nullable=False)
    detected    = Column(Text)
    confidence  = Column(Float)
    clip_path   = Column(String)
    thumbnail   = Column(String)
    ai_summary  = Column(Text)
    # Usa datetime.now (horário local do servidor) em vez de utcnow
    created_at  = Column(DateTime, default=datetime.now)
    reviewed    = Column(Boolean, default=False)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    role       = Column(String, nullable=False)
    content    = Column(Text, nullable=False)
    event_id   = Column(String)
    created_at = Column(DateTime, default=datetime.now)


_engine          = None
_session_factory = None


async def init_db(config: dict):
    global _engine, _session_factory
    db_path = config["database"]["path"]
    _engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    _session_factory = sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)


def get_session() -> AsyncSession:
    return _session_factory()