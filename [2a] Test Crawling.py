import os
from src.crawler_by_call_api.yaml_utils import read_yaml
from src.crawler_by_call_api.crawler_function import crawl_item_ids, crawl_item_detail_by_id, crawl_rating_list_by_item_id, crawl_overall_rating_by_id
from src.crawler_by_call_api.write_csv_utils import write_csv_file
from tqdm import tqdm

api_containers = read_yaml('src/api_containers.yaml')

def test_crawl_item_detail():

    item_detail_title = ["id","name",
                         "category",
                         "brand",
                         "price",
                         "discount",
                         "discount_rate",
                         "image_url",
                         "n_sold",
                         "rank", "day_ago_created", "mota"]



    for category, api_container in api_containers.items():
        id_list_api = api_container[0]
        detail_api = api_container[1]
        rating_list_api = api_container[2]
        id = [121169346]
        detail = crawl_item_detail_by_id(detail_api, id)

        # print(detail)

def test_crawl_item_overall_detail():

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
        id_list_api = api_container[0]
        detail_api = api_container[1]
        rating_list_api = api_container[2]
        id = [121169346]
        overall = crawl_overall_rating_by_id(rating_list_api, id)
        
        print(overall)


def test_crawl_item_comment():

    rating_title = ["user_id",
                    "item_id",
                    "rating",
                    "timestamp",
                    "comment", "username", "purchased_at", "group_id", "joined_time", "total_review", "total_thank"]
    
    for category, api_container in api_containers.items():
        id_list_api = api_container[0]
        detail_api = api_container[1]
        rating_list_api = api_container[2]
        id = [116410110]
        rating_list = crawl_rating_list_by_item_id(
                rating_list_api, id[0])

        print(rating_list[:1])

test_crawl_item_overall_detail()