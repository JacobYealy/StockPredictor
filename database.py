from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class StockData(Base):
    __tablename__ = 'stock_data'

    id = Column(Integer, primary_key=True)
    date = Column(String)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)

engine = create_engine('sqlite:///stock_data.db')
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

def insert_data(data):
    for record in data:
        stock_entry = StockData(date=record[0], open=record[1], high=record[2], low=record[3], close=record[4], volume=record[5])
        session.add(stock_entry)
    session.commit()
