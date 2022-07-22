import datetime
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column,  Integer,String,Float,Text,ForeignKey,DateTime,UniqueConstraint,Index

Base = declarative_base()

class Users(Base):
    __tablename__ = 'users'
    id = Column(Integer,primary_key=True)
    username = Column(String(255),nullable=False)
    phone = Column(String(255),nullable=False)
    email = Column(String(255),nullable=False)
    password = Column(String(255),nullable=False)
    role = Column(String(255),nullable=False)
    create_time = Column(DateTime,default=datetime.datetime.now)

class CellCancer(Base):
    __tablename__ = 'cell_cancer'
    id = Column(Integer,primary_key=True)
    name = Column(String(32),nullable=False)
    thickness = Column(Integer,nullable=False)
    sizeUniformity = Column(Integer,nullable=False)
    shapeUniformity = Column(Integer,nullable=False)
    adhesion = Column(Integer,nullable=False)
    cellSize = Column(Integer,nullable=False)
    nakedCore = Column(Integer,nullable=False)
    chromatinColor = Column(Integer,nullable=False)
    nucleolusCondition = Column(Integer,nullable=False)
    mitosisSituation = Column(Integer,nullable=False)
    disease_rate = Column(Float,nullable=False)
    create_time = Column(DateTime,default=datetime.datetime.now)

    # __table_args__ = (
    #     UniqueConstraint('id', 'name', name='uix_id_name'),  # 联合唯一
    #     Index('ix_id_name', 'name'),  # 索引
    # )
class BloodCancer(Base):
    __tablename__ = 'blood_cancer'
    id = Column(Integer,primary_key=True)
    name = Column(String(32),nullable=False)
    age = Column(Integer, nullable=False)
    bmi = Column(Float,nullable=False)
    glucose = Column(Float,nullable=False)
    insulin = Column(Float,nullable=False)
    homa = Column(Float,nullable=False)
    leptin = Column(Float,nullable=False)
    adiponectin = Column(Float,nullable=False)
    resistin = Column(Float,nullable=False)
    mcp1 = Column(Float,nullable=False)
    disease_rate = Column(Float,nullable=False)
    create_time = Column(DateTime,default=datetime.datetime.now)

def create_table():
    #创建engine对象
    engine = create_engine(
        "mysql+pymysql://root:abc123@127.0.0.1:3306/breast_cancer?charset=utf8",
        max_overflow=0,
        pool_size=5,
        pool_timeout=50,
        pool_recycle=5
    )
    #通过engine对象创建表
    Base.metadata.create_all(engine)

if __name__ == '__main__':
    create_table()