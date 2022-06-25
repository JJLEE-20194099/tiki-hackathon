from mongoengine import *
import datetime

from traitlets import default
class Gifts(Document):
    username = StringField(required=True)
    email = EmailField(required=True)
    nameSender = StringField(required=True)
    phoneSender = StringField(required=True)
    content = StringField(required=True)
    nameReceiver = StringField(required=True)
    phoneReceiver = StringField(required=True)
    total = IntField(required=True)
    address = StringField(required=True)
    items = ListField(required=True)
    status = BooleanField(required=True, default=False)
    createdAt = DateTimeField(default=datetime.datetime.now())

    def to_document(self):
        return self.__dict__

class Outfits(Document):
    userId = ObjectIdField(required=True)
    items = ListField(required=True)
    reviews = IntField(default=0)
    likes = IntField(default=0)
    views = IntField(default=0)
    desc = StringField(default='')
    set_id = StringField(default='')
    date = StringField(default='')
    createdAt = DateTimeField(default=datetime.datetime.now())
    
    def to_document(self):
        return self.__dict__

class OutfitReviews(Document):
    outfitId = ObjectIdField(required=True)
    userId = ObjectIdField(required=True)
    review = StringField()
    name = StringField()
    numOfLikes = IntField(default=0)
    numOfComments = IntField(default=0)
    createdAt = DateTimeField(default=datetime.datetime.now())
    
    def to_document(self):
        return self.__dict__
    

class Users(Document):
    username = StringField(required=True)
    password = StringField(required=True)
    email = EmailField(required=True)

    numOfConnections = IntField(default=0)

    def to_document(self):
        return self.__dict__

