Цель vsco2 - автоматизировать поиск и закачку нюдсов с VSCO с помощью машинного обучения.


Описание:
    - Вкратце, vsco2 сохраняет 20% (или указанный процент) случайных пикч профиля, ищет в них нюдсы.
    Если находит хотя бы один, то сохраняет весь профиль в отдельную папку.
    - vsco2 использует https, в отличие от VSCO-Downloader, который не шифрует траффик и использует http
    - Можно вставить несколько ссылок, скрипт прочекает и загрузит все профили.
    - Используется модель поиска nsfw, которую обучили CNN на 60 ГБ фото.
    - Можно использовать отдельно модифицированный vscoscraper.py, если не нужен ML и просто хочется иметь https

    Пикчи загружает модифицированный https://github.com/mvabdi/vsco-scraper
    За машинное обучние отвечает модифицрованный https://github.com/GantMan/nsfw_model/releases/tag/1.1.0
    (по дефолту он почему-то не хотел работать с обученной моделью, которую они прикладывают)


Запуск:
    python3 vsco.py

Help message generated by argparse:
    usage: vsco2.py [-h] [floor] [sample]

    positional arguments:
      floor       порог для nsfw_index (подробнее в README.txt)
      sample      процент пикч, которые будут предскачаны (подробнее в README.txt)

    optional arguments:
      -h, --help  show this help message and exit

Пример работы:
    Enter link:
    https://vsco.co/evgeniya8888/gallery
    https://vsco.co/aliyahhns22/gallery
    https://vsco.co/darinachasova/gallery
    Finding new posts of evgeniya8888: 0 posts [00:00, ? posts/s]
    19:26:40.03d INFO vsco2 - analyze_profile: Analyzing username: evgeniya8888
    19:26:40.03d WARNING vsco2 - analyze_profile: No images found
    19:26:40.03d INFO vsco2 - analyze_profile: NO NSFW FOUND, pics scanned: 0
    Enter link:
    Finding new posts of aliyahhns22: 60 posts [00:00, 63.80 posts/s]
    19:26:42.03d INFO vsco2 - analyze_profile: Analyzing username: aliyahhns22
    19:26:48.03d INFO vsco2 - analyze_profile: NSFW FOUND, index: 0.9761589970439672, pics scanned: 5, filename: 1613995944.jpg
    Finding new posts of aliyahhns22: 55 posts [00:00, 84.58 posts/s]
    Downloading posts of aliyahhns22:  78%|███████▊  | 43/55 [00:43<00:24,  2.01s/ posts]
    ...


Гайд по установке:
    1. Потребуется Питон, в идеале 3.9+ версии, хер знает, запустится ли на более ранней версии
    2. В консоли перейти в папку с этим проектом (cd ПУТЬ_К_ПРОЕКТУ)
    3. Прописать команду:
        python3 -m pip install -r requirements.txt
    Если система не будет знать, что такое python3, можно попробовать python или указать полный путь до python.exe


Не очень важная информация:
    1. https://github.com/NicholasDawson/VSCO-Downloader - дерьмовая реализация парсера, спизженная у vsco-scraper
    2. mobilenet's indexes: drawings, hentai, neutral, porn, sexy
