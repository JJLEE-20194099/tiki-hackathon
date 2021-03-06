from flask import Flask
from importlib_metadata import metadata
from pymongo import MongoClient
from mongoengine import connect
from flask import request
from flask import jsonify
from flask_jwt_extended import create_access_token
from datetime import timedelta
from flask_jwt_extended import get_jwt_identity
from flask_jwt_extended import jwt_required
from flask_jwt_extended import JWTManager
from flask_cors import CORS
import pandas as pd
import numpy as np
from numpy.linalg import norm
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from bson import ObjectId
from bson.json_util import dumps
import json
from faker import Faker
import random
from models import *
from sklearn.cluster import KMeans
from sklearn import preprocessing
import cv2
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, GlobalMaxPooling2D, Conv2D, UpSampling2D, MaxPooling2D, Dropout, Activation
from tensorflow.keras.models import Model, Sequential, model_from_json
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import GlorotUniform, HeNormal, HeUniform, GlorotNormal
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from utils import *
import PIL.Image
from IPython.display import Image, display
from tensorflow.keras.preprocessing import image

from TikiAIFashionFinding.cloth_detection import Detect_Clothes_and_Crop
from TikiAIFashionFinding.utils_my import Read_Img_2_Tensor, Save_Image, Load_DeepFashion2_Yolov3

import cloudinary
import cloudinary.uploader
import cloudinary.api

import requests

cloudinary.config( 
  cloud_name = "diw1ijbmf", 
  api_key = "887928444761171", 
  api_secret = "25_emp2T4VfYyeGjg27eJOF2pPY" 
)

app = Flask(__name__)
app.debug = True
JWTManager(app)
CORS(app)
app.config["JWT_SECRET_KEY"] = "super-secret"
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=365*24)

# connect(db='tiki_fashion_app', host="mongodb://localhost:27017")
connect(db='fashion', host="mongodb+srv://ddcmtitiki:187691646@cluster0.zbruhcv.mongodb.net/?retryWrites=true&w=majority")

# Model variable
ai_search_model = None

crop_model = None


# Done
@app.route("/signup",methods=["POST"])
def signup():
    req = eval(list(request.form.to_dict().keys())[0])
    username = req['username']
    email = req['email']
    password = req['password']
    re_password = req['re_password']
    if (password != re_password):
        return jsonify({"error": "Nh???p l???i m???t kh???u kh??ng ????ng"})
    
    Users(username=username,password=password,email=email).save()
    return jsonify({"notice": "????ng k?? th??nh c??ng"})

@app.route("/login", methods=["POST"])
def login():
    req = eval(list(request.form.to_dict().keys())[0])
    username = req['username']
    password = req['password']
    user = Users.objects(username=username,password=password).first()
    access_token = create_access_token(identity=user.username)
    return jsonify({"access_token":access_token})

@app.route("/test", methods=["GET"])
@jwt_required()
def test():
    return "Connect Successfully"

@app.route("/contact-reviewer", methods=["POST"])
@jwt_required()
def contactReviwewer():
    username = get_jwt_identity()
    user = Users.objects(username=username).first()
    senderEmail = user.email
    req = eval(list(request.form.to_dict().keys())[0])
    reviewerEmail = req['reviewerEmail']
    content = req['content']

    sender_address = 'tikihackathon@gmail.com'
    sender_pass = 'ahaphjbkojyqurux'
    receiver_address = reviewerEmail

    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = 'Th?? m???i li??n h??? review s???n ph???m'
    mail_content = '''
    Xin ch??o {}

    {}
    Li??n h??? v???i email {} ????? bi???t th??m chi ti???t. C???m ??n b???n r???t nhi???u
    
    {}
    '''.format(receiver_address, content, senderEmail, user.username)
    message.attach(MIMEText(mail_content, 'plain'))
    session = smtplib.SMTP('smtp.gmail.com', 587)
    session.starttls()
    session.login(sender_address, sender_pass)
    text = message.as_string()
    session.sendmail(sender_address, receiver_address, text)
    session.quit()

    reviewer = Users.objects(email = reviewerEmail).first()
    numOfConnections = reviewer.numOfConnections + 1
    reviewer.update(numOfConnections=numOfConnections)
    return jsonify({"notice":"B???n li??n h??? th??nh c??ng"})




@app.route("/serve-gift", methods=["POST"])
@jwt_required()
def doGiftService():
    req = eval(list(request.form.to_dict().keys())[0])
    username = req['username']
    email = req['email']
    nameSender = req['nameSender']
    phoneSender = req['phoneSender']
    content = req['content']
    nameReceiver = req['nameReceiver']
    phoneReceiver = req['phoneReceiver']
    total = req['total']
    address = req['address']
    items = req['items']
    
    Gifts(username=username, email=email, nameSender=nameSender, phoneSender=phoneSender, content=content, nameReceiver=nameReceiver, phoneReceiver=phoneReceiver, total=total, address=address, items=items).save()

    sender_address = 'tikihackathon@gmail.com'
    sender_pass = 'ahaphjbkojyqurux'
    receiver_address = email

    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = 'D???ch v??? ?????t qu?? TIKI'
    mail_content = '''Xin ch??o {}
    Ch??c m???ng b???n ???? ?????t qu?? th??nh c??ng t??? h??? th???ng TIKI c???a ch??ng t??i
    Thank You'''.format(nameSender)
    message.attach(MIMEText(mail_content, 'plain'))
    session = smtplib.SMTP('smtp.gmail.com', 587)
    session.starttls()
    session.login(sender_address, sender_pass)
    text = message.as_string()
    session.sendmail(sender_address, receiver_address, text)
    session.quit()
    print('Mail Sent')
    return jsonify({"notice":"B???n ???? ?????t d???ch v??? t???ng qu?? th??nh c??ng, xin vui l??ng ki???m tra l???i qu?? trong email:{} c???a b???n".format(email)})

def get_outfit_suggestion(predicted, metadata_imgs, threshold):
    res = []
    for i, value in enumerate(predicted):
        if value >= threshold:
            res.append(metadata_imgs[i])
    return res

def convert_2_vi_name(name):
    class_names = ['short_sleeve_top', 'long_sleeve_top', 'short_sleeve_outwear', 'long_sleeve_outwear',
                  'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeve_dress',
                  'long_sleeve_dress', 'vest_dress', 'sling_dress']

    if name == 'short_sleeve_top':
        return "??o ph??ng ng???n tay (short sleeve top)"
    
    if name == "long_sleeve_top":
        return "??o ph??ng d??i tay (long sleeve top)"
    
    if name == "short_sleeve_outwear":
        return "??o kho??c ng???n tay (short sleeve outwear)"
    
    if name == "long_sleeve_outwear":
        return "??o kho??c d??i tay (long sleeve outwear)"
    
    if name == "vest":
        return "A?? ba l??? nam (vest)"
    if name == "sling":
        return "??o y???m (sling)"
    
    if name == "shorts":
        return "Qu???n ????i (shorts)"
    
    if name == "trousers":
        return "Qu???n d??i (trousers)"
    
    if name == "skirt":
        return "V??y ng???n (skirt)"
    
    if name == "short_sleeve_dress":
        return "V??y d??i ng???n tay (short sleeve dress)"
    
    if name == "long_sleeve_dress":
        return "V??y d??i d??i tay (long sleeve dress)"
    
    if name == "vest_dress":
        return "V??y d??i kh??ng c?? ???ng tay (vest dress)"
    
    if name == "sling_dress":
        return "V??y y???m (sling dress)"
 

def crop_image(url):
    im = PIL.Image.open(requests.get(url, stream=True).raw)
    img_arr = np.asarray(im)
    image_path = './temporary_save_img.jpg'
    cv2.imwrite(image_path, img_arr)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = Read_Img_2_Tensor(image_path)
    res_list = Detect_Clothes_and_Crop(img_tensor, crop_model)

    image_base_path = './temporary_save'
    if os.path.exists(image_base_path) == False:
        os.makedirs(image_base_path)
    
    data = dict()
    image_crop_path_list = []
    cate_list = []
    for res in res_list:
        img_crop = res[0]
        path = '{}/{}.jpg'.format(image_base_path, res[1])
        image_crop_path_list.append(path)
        cate_list.append(res[1])
        cv2.imwrite(path, img_crop*255)
    
    return [image_crop_path_list, cate_list]

def preprocess(image_path, extract_model):
    if (image_path is None):
        return np.zeros((2048, ))
    features = extract_img_features(image_path, extract_model)
    return features

def process_empty_cloth(cloth_list):
  if len(cloth_list) == 0:
    return [None]
  return cloth_list
def concat_vec(outfit):
  res = []
  for cloth in outfit:
    res = res + list(cloth)
  return np.array(res)

@app.route("/suggest-outfit", methods=["POST"])
def suggestOutfit():
    weights_file_path = './weights/tiki_deep_outfit_suggestion_weights.h5'
    json_file = open('./json_model/tiki_deep_outfit_suggestion_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    tiki_fashion_model = model_from_json(loaded_model_json)
    tiki_fashion_model.load_weights(weights_file_path)
    
    req = eval(list(request.form.to_dict().keys())[0])
    upload_imgs_arr_list = req['items']

    preprocess_outfit = []
    tops = process_empty_cloth(upload_imgs_arr_list["top"])
    pullovers = process_empty_cloth(upload_imgs_arr_list["pullover"])
    outerwears = process_empty_cloth(upload_imgs_arr_list["outerwear"])
    bottoms = process_empty_cloth(upload_imgs_arr_list["bottom"])
    shoes = process_empty_cloth(upload_imgs_arr_list["shoe"])
    bags = process_empty_cloth(upload_imgs_arr_list["bag"])
    dresses = process_empty_cloth(upload_imgs_arr_list["dress"])



    candidates = []
    outfit_paths = []

    global ai_search_model
    if ai_search_model is None:
        ai_search_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        ai_search_model.trainable = False
        ai_search_model = Sequential([ai_search_model, GlobalMaxPooling2D()])

    for top in tqdm(tops):
        for pullover in pullovers:
            for outerwear in outerwears:
                for bottom in bottoms:
                    for shoe in shoes:
                        for bag in bags:
                            for dress in dresses:
                                top_feature = preprocess(top, ai_search_model)
                                pullover_feature = preprocess(pullover, ai_search_model)
                                outerwear_feature = preprocess(outerwear, ai_search_model)
                                bottom_feature = preprocess(bottom, ai_search_model)
                                shoe_feature = preprocess(shoe, ai_search_model)
                                bag_feature = preprocess(bag, ai_search_model)
                                dress_feature = preprocess(dress, ai_search_model)
                                candidates.append([top_feature, pullover_feature, outerwear_feature, bottom_feature, shoe_feature, bag_feature, dress_feature])
                                outfit_paths.append([top, pullover, outerwear, bottom, shoe, bag, dress])


    

                                    
    candidates = np.array(candidates)
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
    X = [concat_vec(candidate) for candidate  in candidates]
    kmeans.fit(np.array(X))
    labels = kmeans.labels_
    suggest_dict = dict()

    for i, label in enumerate(labels):
        if str(label) not in suggest_dict.keys():
            suggest_dict.update({str(label): {
                "cans": [candidates[i]],
                "paths": [outfit_paths[i]]
            }})
        else:
            suggest_dict[str(label)]["cans"].append(candidates[i])
            suggest_dict[str(label)]["paths"].append(outfit_paths[i])
    
    suggest_res = []
    for key in suggest_dict.keys():
        cans = np.array(suggest_dict[key]["cans"])
        predict_values = np.array(tiki_fashion_model.predict([cans[:, 0], cans[:, 1], cans[:, 2], cans[:, 3], cans[:, 4], cans[:, 5], cans[:, 6]]))
        paths = suggest_dict[key]["paths"]
        suggest_res.append(paths[np.argmax(predict_values)])

    return jsonify({"outfit_suggestion_list": suggest_res})

# Done
@app.route("/create-outfit", methods=["POST"])
@jwt_required()
def createOutfit():
    username = get_jwt_identity()
    userId = ObjectId(Users.objects(username=username).first().pk)
    req = eval(list(request.form.to_dict().keys())[0])
    # outfit = request.json.get("outfit", None)
    items = []
    for i in req['items'].keys():
        if len(req['items'][i]) > 0:
            items.append({
                'name': i,
                'image': req['items'][i][0],
                'likes': 0,
                'price': random.randint(1, 100)
            })
    # items = req['items']
    desc = req["desc"]
    Outfits(userId=userId, items=items, desc=desc).save()
    return jsonify({"notice":"B???n ???? t???o outfit th??nh c??ng"})



#Done
@app.route("/generate-user", methods=["POST"])
def generateUser():
    fake = Faker()
    for i in range(100):
        username = fake.pystr(min_chars=None, max_chars=10)
        email = fake.pystr(min_chars=None, max_chars=20) + "@gmail.com"
        password = "123"
        Users(username=username,password=password,email=email).save()
    
    return jsonify({"notice":"Fake users th??nh c??ng"})

# Done
@app.route("/generate-outfit/", methods=["POST"])
def generateOutfit():

    outfit_list = json.load(open('./fashion_metadata/train_no_dup.json'))
    users = Users.objects()
    num_users = len(users)

    for outfit in outfit_list:
        idx = random.randint(0, num_users-1)
        userId = users[idx].pk
        items = outfit["items"]
        reviews = int(outfit["views"]/10)
        likes = outfit["likes"]
        views = outfit["views"]
        desc = outfit["desc"]
        set_id = outfit["set_id"]
        date = outfit["date"]

        Outfits(userId=userId, items=items, reviews=reviews, likes=likes, views=views, desc=desc, set_id=set_id, date=date).save()
    
    return jsonify({"notice":"Fake outfits th??nh c??ng"})


# Done
@app.route("/get-category", methods=["GET"])
def getCategory():
    with open("./fashion_metadata/category_id.txt", "r") as f:
        cates = f.readlines()
        cates = [cate.strip() for cate in cates]
        category_id = request.args.get("category_id")
        for cate in cates:
            cate_id = cate.split(" ")[0]
            if cate_id == category_id:
                return jsonify({"notice":"Fake category th??nh c??ng", "data": cate.split(" - ")[-1]})

# Done
@app.route("/get-outfit", methods=['GET'])
def getOutfit():
    outfitId = ObjectId(request.args.get("outfit_id"))
    outfit = Outfits.objects.get(pk=outfitId)
    return jsonify({"notice":"L???y ra outfit th??nh c??ng", "data": json.loads(outfit.to_json())})

# Done
@app.route("/get-my-outfit", methods=['GET'])
@jwt_required()
def getMyOutfit():
    username = get_jwt_identity()
    userId = ObjectId(Users.objects(username=username).first().pk)
    outfits = Outfits.objects(userId=ObjectId(userId))
    return jsonify({"notice":"L???y ra outfit th??nh c??ng", "data": json.loads(outfits.to_json())})

@app.route("/get-outfit-by-id", methods=['GET'])
@jwt_required()
def getOutfitByUserId():
    userId = ObjectId(request.args.get("userId"))
    outfits = Outfits.objects(userId=ObjectId(userId))
    return jsonify({"notice":"L???y ra outfit th??nh c??ng", "data": json.loads(outfits.to_json())})

# Done
@app.route("/get-all-outfit", methods=['GET'])
@jwt_required()
def getAllOutfit():
    type = request.args.get("type")
    if type == 'reviews':
        outfits = Outfits.objects.order_by('-reviews', '-likes', '-views')[:10]
        return jsonify({"notice":"L???y ra outfit th??nh c??ng", "data": json.loads(outfits.to_json())})
    if type == 'views':
        outfits = Outfits.objects.order_by('-views', '-likes', '-reviews')[:10]
        return jsonify({"notice":"L???y ra outfit th??nh c??ng", "data": json.loads(outfits.to_json())})
    outfits = Outfits.objects.order_by('-likes', '-reviews', '-views')[:10]
    return jsonify({"notice":"L???y ra outfit th??nh c??ng", "data": json.loads(outfits.to_json())})

# Done
@app.route("/get-all-reviewer", methods=["GET"])
@jwt_required()
def getAllReviewer():
    reviewers = OutfitReviews.objects.aggregate([{
        "$group": {"_id": "$userId", "numOfLikes": {"$sum": "$numOfLikes"}, "numOfComments": {"$sum": "$numOfComments"}, "outfitIdList": {"$push": "$outfitId"}}
    }])
    reviewersJson = json.loads(dumps(reviewers))
    data = []
    for reviewer in reviewersJson:
        reviewer["numOfReviewedOutfit"] = len(reviewer["outfitIdList"])
        data.append(reviewer)
    
    return jsonify({"notice":"L???y ra reviewer th??nh c??ng", "data": data})

# Done
@app.route("/get-user-by-id", methods=["GET"])
def getUserById():
    user = Users.objects(pk=ObjectId(request.args.get("userId"))).first()
    return jsonify({"notice":"L???y ra user th??nh c??ng", "data": json.loads(user.to_json())})

# Done
@app.route("/get-all-review-outfit", methods=['GET'])
@jwt_required()
def getAllReviewOutfit():
    reviewOutfits = OutfitReviews.objects().all()
    return jsonify({"notice":"L???y ra review outfit th??nh c??ng", "data": json.loads(reviewOutfits.to_json())})

# Done
@app.route("/review-outfit", methods=["POST"])
@jwt_required()
def reviewOutfit():
    username = get_jwt_identity()
    req = eval(list(request.form.to_dict().keys())[0])
    userId = ObjectId(Users.objects(username=username).first().pk)
    outfitId = ObjectId(req['outfitId'])
    review = req['review']

    OutfitReviews(userId=userId, outfitId=outfitId, review=review).save()
    outfit = Outfits.objects.get(pk=outfitId)
    numOfReviews = outfit.reviews + 1
    numOfViews = outfit.views + 1
    Outfits.objects.get(pk=outfitId).update(reviews=numOfReviews)
    Outfits.objects.get(pk=outfitId).update(views=numOfViews)

    return jsonify({"notice":"B???n review th??nh c??ng"})

#Done
@app.route("/like-review-outfit", methods=["POST"])
@jwt_required()
def likeReviewOutfit():
    username = get_jwt_identity()
    userId = ObjectId(Users.objects(username=username).first().pk)
    reviewOutfitId = ObjectId(request.json.get("outfitId", None))

    outfitReview = OutfitReviews.objects.get(pk=reviewOutfitId)
    outfitId = outfitReview.pk
    numOfLikes = outfitReview.numOfLikes
    outfitReview.update(numOfLikes=numOfLikes+1)

    outfit = Outfits.objects.get(pk=outfitId)
    numOfLikes = outfit.likes + 1
    numOfViews = outfit.views + 1
    outfit.update(likes=numOfLikes)
    outfit.update(views=numOfViews) 

    return jsonify({"notice":"B???n like b??i vi???t th??nh c??ng"})
def convert_image_url_to_array(url):
  return np.array(PIL.Image.open(requests.get(url, stream=True).raw).resize((224, 224)))

def extract_img_features(img_path, model):
  img_array = convert_image_url_to_array(img_path)
  expand_img = np.expand_dims(img_array,axis=0)
  preprocessed_img = preprocess_input(expand_img)
  result_to_resnet = model.predict(preprocessed_img)
  flatten_result = result_to_resnet.flatten()
  # normalizing
  result_normlized = flatten_result / norm(flatten_result)
  return result_normlized

def extract_crop_img_features(img_path, model):
    img = image.load_img(img_path,target_size=(224,224))
    img_arr = image.img_to_array(img)
    expand_img = np.expand_dims(img_arr,axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    result_normlized = flatten_result / norm(flatten_result)
    return result_normlized

def recommend(features, features_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)

    distence, indices = neighbors.kneighbors([features])

    return indices


@app.route("/ai-search", methods=["POST"])
def searchByAI():
    req = eval(list(request.form.to_dict().keys())[0])
    uploaded_file = req['uploaded_file']
    im = PIL.Image.open(requests.get(uploaded_file, stream=True).raw).resize((224, 224))
    size = (400, 400)
    resized_im = im.resize(size)
    # extract features of uploaded image
    features_list = pickle.load(open("./image_features_embedding.pkl", "rb"))
    img_files_list = pickle.load(open("./img_files.pkl", "rb"))

    global crop_model
    if crop_model is None:
        crop_model = Load_DeepFashion2_Yolov3()

    
    
    detect_cloth_list = crop_image(uploaded_file)

    crop_image_path_list = detect_cloth_list[0]
    cate_list = detect_cloth_list[1]
    

    global ai_search_model
    if ai_search_model is None:
        ai_search_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        ai_search_model.trainable = False
        ai_search_model = Sequential([ai_search_model, GlobalMaxPooling2D()])
    
    
    data = dict()
    for _ in range(len(crop_image_path_list)):
        crop_image_path = crop_image_path_list[_]
        cate = cate_list[_]
        features = extract_crop_img_features(crop_image_path, ai_search_model)
        img_indicess = recommend(features, features_list)
        paths = []
        for i in range(len(img_indicess[0])):
            path = img_files_list[img_indicess[0][i]]
            paths.append(path)
        data.update({convert_2_vi_name(cate): paths})

    return jsonify({"notice":"B???n ???? t??m ki???m b???ng AI th??nh c??ng", "data": data})

if __name__ == "__main__":    
    app.run()
