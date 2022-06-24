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
# os.makedirs(OUTPUT, exist_ok=False)
# os.makedirs(CATEGORY_OUTPUT, exist_ok=False)


def craw():

    item_detail_title = ["id","name",
                         "category",
                         "brand",
                         "price",
                         "discount",
                         "discount_rate",
                         "image_url",
                         "n_sold",
                         "rank", "day_ago_created", "mota"]

    item_overall_title = ["id",
                          "avg_rating",
                          "n_reviews",
                          "n_rate_5",
                          "n_rate_4",
                          "n_rate_3",
                          "n_rate_2",
                          "n_rate_1",
                          "rate_with_img"]


    for category, api_container in api_containers.items():
        category_dir = os.path.join(CATEGORY_OUTPUT, category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)


        id_list_api = api_container[0]
        detail_api = api_container[1]
        rating_list_api = api_container[2]

        item_id_list_path = os.path.join(category_dir, 'item_ids.txt')
        item_detail_list_path = os.path.join(category_dir, 'item_detail.csv')
        item_overall_list_path = os.path.join(category_dir, 'item_overall.csv')

        item_id_list = crawl_item_ids(id_list_api)
        write_csv_file(item_id_list, item_id_list_path, 'w')

        write_csv_file([item_detail_title], item_detail_list_path, 'w')
        for id in item_id_list:
            detail = crawl_item_detail_by_id(detail_api, id)
            write_csv_file([detail], item_detail_list_path, 'a')
        
        write_csv_file([item_overall_title], item_overall_list_path, 'w')
        for id in item_id_list:
            overall = crawl_overall_rating_by_id(rating_list_api, id)
            write_csv_file([overall], item_overall_list_path, 'a')


craw()