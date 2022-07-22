# -*- coding:utf-8 -*-
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

import train
import train1
from model import CellCancer,BloodCancer
from sqlalchemy.orm import scoped_session
from model import Users
# from user import *

from flask import make_response,Flask,request,session,jsonify
from flask_cors import CORS, cross_origin
from utility import ImageCode
import os


app = Flask(__name__)
# app.secret_key = "ajfoigjdhdgjglhiuwq"
app.config['SECRET_KEY'] = os.urandom(24)
CORS(app, supports_credentials=True)



engine = create_engine(
        "mysql+pymysql://root:abc123@127.0.0.1:3306/breast_cancer?charset=utf8",
        max_overflow=0,
        pool_size=5,
    )

#制造一个session对象
Session1 = sessionmaker(bind=engine)
session1 = scoped_session(Session1)
# session = Session()


@app.route('/vcode',methods=["GET"])
@cross_origin(supports_credentials=True)
def genCode():
    code,bstring = ImageCode().get_code()
    response = make_response(bstring)
    response.headers['Content-Type'] = 'image/png'
    session['vcode'] = code.lower()
    print('session-->',session)
    session.permanent = True
    return response

# 登录
@app.route("/login",methods=["POST"])
@cross_origin(supports_credentials=True)
def login():
    phone = request.json.get("phone").strip()
    password = request.json.get("password").strip()
    vcode = request.json.get("vcode").lower().strip()
    print('session-login-->', session)
    print('vcode---->', vcode)
    if vcode == '':
        return 'vcode-error'
    else:
        res = session1.query(Users).filter_by(phone=phone,password=password).first()
        if(res.phone != '') and (res.password != ''):
            result1 = {'status': 200, 'msg': '登录成功',
                       'data': {
                           'username':res.username,
                           'phone':res.phone,
                           'email':res.email,
                           'password':res.password,
                           'role': res.role}
                       }
            return jsonify(result1)
        else:
            result2 = {'status': 400, 'msg': '登录失败'}
            return jsonify(result2)


# 注册
@app.route("/register",methods=["POST"])
@cross_origin(supports_credentials=True)
def register():
    username = request.json.get("username").strip()
    phone = request.json.get("phone").strip()
    email = request.json.get("email").strip()
    password = request.json.get("password").strip()
    role = request.json.get("role").strip()

    res1 = session1.query(Users).filter_by(username=username).all()
    res2 = session1.query(Users).filter_by(phone=phone).all()
    res3 = session1.query(Users).filter_by(email=email).all()
    if len(res1)>0:
        return 'username-failed'
    elif len(res2)>0:
        return 'phone-failed'
    elif len(res3)>0:
        return 'email-failed'
    elif len(res1)== 0 and len(res2)==0 and len(res3)==0 and username!='' and phone!='' and email!='' and password!=''  and role!='':
        obj = Users(
            username=username,
            phone=phone,
            email=email,
            password=password,
            role=role
        )
        # 添加数据
        session1.add(obj)
        session1.commit()
        session1.close()
        return '注册成功'
    else:
        return 'failed register'


#细胞学分析
@app.route("/", methods=["POST"])
@cross_origin(supports_credentials=True)
def cellAnalysis():
    patientName = request.json.get("patientName").strip()
    thickness = int(request.json.get("thickness").strip())
    sizeUniformity = int(request.json.get("sizeUniformity").strip())
    shapeUniformity = int(request.json.get("shapeUniformity").strip())
    adhesion = int(request.json.get("adhesion").strip())
    cellSize = int(request.json.get("cellSize").strip())
    nakedCore = int(request.json.get("nakedCore").strip())
    chromatinColor = int(request.json.get("chromatinColor").strip())
    nucleolusCondition = int(request.json.get("nucleolusCondition").strip())
    mitosisSituation = int(request.json.get("mitosisSituation").strip())
    a = [
        thickness, sizeUniformity, shapeUniformity, adhesion, cellSize,
        nakedCore, chromatinColor, nucleolusCondition, mitosisSituation
    ]
    disease_rate=train.new_prediction(a)
    obj = CellCancer(
        name=patientName,
        thickness=thickness,
        sizeUniformity=sizeUniformity,
        shapeUniformity=shapeUniformity,
        adhesion=adhesion,
        cellSize=cellSize,
        nakedCore=nakedCore,
        chromatinColor=chromatinColor,
        nucleolusCondition=nucleolusCondition,
        mitosisSituation=mitosisSituation,
        disease_rate=disease_rate
    )
    # 添加数据
    session1.add(obj)
    session1.commit()
    session1.close()

    return str(disease_rate)


@app.route("/analysis",methods=["GET"])
@cross_origin(supports_credentials=True)
def fetchCellList():
    # 查询数据
    # res = session.query(CellCancer).filter_by(id=1)
    res = session1.query(CellCancer).all()
    list = []
    for row in res:
        dict = {}
        for k, v in row.__dict__.items():
            if not k.startswith('_sa_instance_state'):
                dict[k] = v
        list.append(dict)
    print('查询结果', list)
    session1.commit()
    session1.close()
    return jsonify(list)

@app.route("/delete/<id>",methods=["POST"])
@cross_origin(supports_credentials=True)
def deleteList(id):
    # 查询数据
    res = session1.query(CellCancer).filter_by(id=id).delete()
    result = session1.query(CellCancer).all()
    list = []
    for row in result:
        dict = {}
        for k, v in row.__dict__.items():
            if not k.startswith('_sa_instance_state'):
                dict[k] = v
        list.append(dict)
    session1.commit()
    session1.close()
    return jsonify(list)

# 血常规分析
@app.route("/blood", methods=["POST"])
@cross_origin(supports_credentials=True)
def bloodAnalysis():
    patientName = request.json.get("patientName").strip()
    age = int(request.json.get("age").strip())
    bmi = float(request.json.get("bmi").strip())
    glucose = float(request.json.get("glucose").strip())
    insulin = float(request.json.get("insulin").strip())
    homa = float(request.json.get("homa").strip())
    leptin = float(request.json.get("leptin").strip())
    adiponectin = float(request.json.get("adiponectin").strip())
    resistin = float(request.json.get("resistin").strip())
    mcp1 = float(request.json.get("mcp1").strip())
    b = [age,bmi,glucose,insulin,homa,leptin,adiponectin,resistin,mcp1]
    disease_rate = train1.new_prediction(b)
    obj1 = BloodCancer(
        name=patientName,
        age=age,
        bmi=bmi,
        glucose=glucose,
        insulin=insulin,
        homa=homa,
        leptin=leptin,
        adiponectin=adiponectin,
        resistin=resistin,
        mcp1=mcp1,
        disease_rate=disease_rate
    )
    # 添加数据
    session1.add(obj1)
    session1.commit()
    session1.close()
    return str(disease_rate)

@app.route("/bloodlist",methods=["GET"])
@cross_origin(supports_credentials=True)
def fetchBloodList():
    # 查询数据
    res = session1.query(BloodCancer).all()
    list = []
    for row in res:
        dict = {}
        for k, v in row.__dict__.items():
            if not k.startswith('_sa_instance_state'):
                dict[k] = v
        list.append(dict)
    session1.commit()
    session1.close()
    return jsonify(list)

@app.route("/deleteblood/<id>",methods=["POST"])
@cross_origin(supports_credentials=True)
def deleteBloodList(id):
    # 查询数据
    res = session1.query(BloodCancer).filter_by(id=id).delete()
    result = session1.query(BloodCancer).all()
    list = []
    for row in result:
        dict = {}
        for k, v in row.__dict__.items():
            if not k.startswith('_sa_instance_state'):
                dict[k] = v
        list.append(dict)
    session1.commit()
    session1.close()
    return jsonify(list)

if __name__ == '__main__':
    # train
    # train1
    # app.register_blueprint(user)
    app.run(host='127.0.0.4', port=8888, debug=True)
