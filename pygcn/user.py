# from flask import Blueprint,make_response,session,Flask,request,jsonify
# from flask_cors import CORS, cross_origin
# from utility import ImageCode
# from model import Users
#
# user = Blueprint('user',__name__)
# CORS(user, supports_credentials=True)
#
# @user.route('/vcode',methods=["GET"])
# @cross_origin(supports_credentials=True)
# def genCode():
#     code,bstring = ImageCode().get_code()
#     response = make_response(bstring)
#     response.headers['Content-Type'] = 'image/png'
#     print('lower--->', code.lower())
#     session['vcode1'] = code.lower()
#     print(session['vcode1'])
#     return response
#
# # 登录
# @user.route("/login",methods=["POST"])
# @cross_origin(supports_credentials=True)
# def login():
#     phone = request.json.get("phone").strip()
#     password = request.json.get("password").strip()
#     vcode = request.json.get("vcode").lower().strip()
#     print('vcode---->', vcode)
#     print('mycode--->', session.get('vcode1'))
#     if vcode != session.get('vcode1'):
#         return 'vcode-error'
#     else:
#         res = session.query(Users).filter_by(phone=phone,password=password)
#         if len(res)==1 and res[0].phone == phone and res[0].password == password:
#             response = res
#             result1 = {'status': 200, 'msg': '登录成功','response': res}
#             return jsonify(result1)
#         else:
#             result2 = {'status': 400, 'msg': '登录失败'}
#             return jsonify(result2)
