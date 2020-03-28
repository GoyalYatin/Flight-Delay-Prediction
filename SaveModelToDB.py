# -*- coding: utf-8 -*-
"""
Created on Wed May 15 00:34:38 2019

@author: bsoni
"""

import pickle

def save_model_to_db(model, client, db, dbconnection, model_name):
    #pickling the model
    pickled_model = pickle.dumps(model)
    
    #saving model to mongoDB
    # creating connection
    myclient = pymongo.MongoClient(client)
    
    #creating database in mongodb
    mydb = myclient[db]
    
    #creating collection
    mycon = mydb[dbconnection]
    info = mycon.insert_one({model_name: pickled_model, 'name': model_name, 'created_time':time.time()})
    print(info.inserted_id, ' saved with this id successfully!')
    
    details = {
        'inserted_id':info.inserted_id,
        'model_name':model_name,
        'created_time':time.time()
    }
    
    return details

details = save_model_to_db(model = clf, client ='mongodb://localhost:27017/', db = 'models', 
                 dbconnection = 'regression', model_name = 'delay_1')