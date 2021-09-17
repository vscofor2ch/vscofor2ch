import argparse
import random
import logging
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import nsfw_model
import vscoscrape


def calc_nsfw_index(classification: dict) -> float:
    """ {'drawings': 0.04, 'hentai': 0.03, 'neutral': 0.47, 'porn': 0.42, 'sexy': 0.02}  ->  0.48 """
    return sum((classification['hentai'], classification['porn'], classification['sexy']))


def parse_profile_name(profile: str) -> str:
    if 'vsco.co/' not in profile:
        return profile
    return profile.partition('vsco.co/')[2].partition('/')[0]


def is_image(filename: str) -> bool:
    return filename.lower().endswith(('.png', '.jpg', '.jpeg'))


def get_vsco_file_name(file_info: list) -> str:
    return file_info[1] + '.' + file_info[0].rpartition('.')[2]


def analyze_profile(model, profile: str, sample_proportion: float = 0.2, floor: float = 0.75):
    started_dir = os.getcwd()
    try:
        scraper = vscoscrape.Scraper(parse_profile_name(profile))
        scraper.getImageList()
        logging.info(f"Analyzing username: {scraper.username}")
        images = list(filter(lambda info: is_image(info[0]), scraper.imagelist))
        if len(images) == 0:
            logging.warning('No images found')
        for i, image_info in enumerate(random.sample(images, int(len(images) * sample_proportion))):
            scraper.download_img_normal(image_info)
            classification = list(nsfw_model.classify(model,
                                                      f"{scraper.path}/{get_vsco_file_name(image_info)}").values())[0]
            nsfw_index = calc_nsfw_index(classification)
            logging.debug(f"NSFW index: {nsfw_index}, filename: {get_vsco_file_name(image_info)}")
            if nsfw_index > floor:
                logging.info(f"NSFW FOUND, index: {nsfw_index}, "
                             f"pics scanned: {i + 1}, "
                             f"filename: {get_vsco_file_name(image_info)}")
                scraper.getImages()
                return
        else:
            logging.info(f"NO NSFW FOUND, "
                         f"pics scanned: {int(len(images) * sample_proportion)}")
            if os.getcwd() != started_dir:
                directory = os.getcwd()
                os.chdir(started_dir)
                os.rmdir(directory)
    except Exception as e:
        logging.error(e)
        raise
    finally:
        if os.getcwd() != started_dir:
            os.chdir(started_dir)


def parse_args(args: list):
    parser = argparse.ArgumentParser()
    parser.add_argument('floor', nargs='?', type=float, default=0.8,
                        help='порог для nsfw_index (подробнее в README.txt)')
    parser.add_argument('sample', nargs='?', type=float, default=0.2,
                        help='процент пикч, которые будут предскачаны (подробнее в README.txt)')
    return parser.parse_args(args)


if __name__ == '__main__':
    sys.stderr = sys.stdout

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s.%(msecs)d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                        datefmt='%H:%M:%S')

    parsed_args = parse_args(sys.argv[1:])

    model = nsfw_model.load_model('./model_mobilenet_v2_140_224.h5')

    while True:
        link = input('Enter link:\n')
        try:
            analyze_profile(model, link, parsed_args.sample, parsed_args.floor)
        except Exception as e:
            logging.error(e)
