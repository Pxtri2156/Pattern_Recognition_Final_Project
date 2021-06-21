import os 
import yaml
from easydict import EasyDict as edict
from utils.config_parse import get_config


cfg = get_config()
cfg.merge_from_file("E:\Courses\Recognition\Final_Project\Pattern_Recognition_Final_Project\configs\methodA\A.yaml")
cfg.merge_from_file("E:\Courses\Recognition\Final_Project\Pattern_Recognition_Final_Project\configs\methodA\AA.yaml")
dic = {'TRI':"2", 'NGON': "4"}
cfg.merge_from_dict(dic)
print('config A: ', cfg.A.NAME)
print("config B: ", cfg.AA.INFO)
print('config dic: ', cfg.TRI)



