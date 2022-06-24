import os
from src.crawler_by_call_api.yaml_utils import read_yaml
from src.crawler_by_call_api.crawler_function import crawl_item_ids, crawl_item_detail_by_id, crawl_rating_list_by_item_id, crawl_overall_rating_by_id
from src.crawler_by_call_api.write_csv_utils import write_csv_file
from tqdm import tqdm

api_containers = read_yaml('src/api_containers.yaml')

item_ids_file = '../data/item_ids.txt'
item_detail_list_file = '../data/item.txt'
rating_list_file = '../data/rating.csv'

OUTPUT = 'data/crawl_data'
CATEGORY_OUTPUT = 'data/crawl_data/categories'

def craw():
    rating_title = ["user_id",
                    "item_id",
                    "rating",
                    "timestamp",
                    "comment", "username", "purchased_at", "group_id", "joined_time", "total_review", "total_thank"]

    for category, api_container in api_containers.items():
        category_dir = os.path.join(CATEGORY_OUTPUT, category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)

        id_list_api = api_container[0]        
        rating_list_api = api_container[2]

        item_rating_list_path = os.path.join(category_dir, 'item_rating.csv')
        item_id_list = crawl_item_ids(id_list_api)
       
        write_csv_file([rating_title], item_rating_list_path, 'w')
        for id in item_id_list:
            rating_list = crawl_rating_list_by_item_id(
                rating_list_api, id[0])
        
            write_csv_file(rating_list, item_rating_list_path, 'a')

craw()