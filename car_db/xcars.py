#!/usr/bin/env python3

import codecs
import os
import sys
import time
import random
import re
import requests
import json


_pages_dir = 'xcars_pages'
if not os.path.exists(_pages_dir):
    os.makedirs(_pages_dir)
os.chdir(_pages_dir)


_domain = 'https://newcar.xcar.com.cn'


def _is_empty(filename):
    return os.stat(filename).st_size == 0


def _read_from_cache(filename):
    if os.path.exists(filename) and not _is_empty(filename):
        print(f'{filename} exists, using cache')
        with open(filename, 'r') as f:
            return json.load(f)
    return None


def _write_to_cache(filename, ret):
    with codecs.open(filename, 'w', 'utf-8') as f:
        json.dump(ret, f)


def _fetch_page(url, filename):
    if os.path.exists(filename):
        print(f'{filename} exists, using cache')
        with codecs.open(filename, 'r', 'utf-8') as f:
            content = f.read()
    else:
        time.sleep(random.randint(3, 8))
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50'}
        r = requests.get(url, headers=headers)
        assert r.status_code == 200, f'{url} failed with {r}'
        content = r.content.decode('gb2312', 'ignore')
        with codecs.open(filename, 'w', 'utf-8') as f:
            f.write(content)
    return content


def _get_all_brands():
    brands_json_filename = 'brands.json'
    ret = _read_from_cache(brands_json_filename)
    if not ret:
        index_page = _fetch_page(f'{_domain}/car/', 'index.html')
        pb_list = [i for i in index_page.splitlines() if 'var pb_list' in i]
        assert pb_list, f'index.html failed with matching'
        pb_line = pb_list[0]
        brands = re.findall(r'<a href="/car/([^/]+?)/".+?</span>(.+?)</a>', pb_line)
        assert brands, f'index.html failed with finding brands'

        ret = [{'url': f'{_domain}/car/{id_}', 'id': id_, 'name': name.strip()} for id_, name in brands]

        _write_to_cache(brands_json_filename, ret)
    return ret


def _get_all_series_by(brand):
    serie_json_filename = f's_{brand["id"]}.json'
    ret = _read_from_cache(serie_json_filename)
    if not ret:
        s_page = _fetch_page(brand['url'], f's_{brand["id"]}.html')
        series = re.findall(r'<div class="title"><a href="/(\d+)/".+?title="([^"]+)">', s_page)
        assert series, f's_{brand["id"]}.html failed with findall'

        ret = [{'url': f'{_domain}/{id_}/config.htm', 'id': id_, 'name': name.strip()} for id_, name in series]

        _write_to_cache(serie_json_filename, ret)
    return ret


def _get_all_models_by(serie):
    model_json_filename = f'm_{serie["id"]}.json'
    ret = _read_from_cache(model_json_filename)
    if not ret:
        models_page = _fetch_page(serie['url'], f'm_{serie["id"]}.html')
        models = set(re.findall(r'<th ci="(\d+)" style="', models_page))
        if models:
            ret = list(models)
        else:
            print(f'{serie["name"]}({serie["id"]}) is empty')
            ret = []

        _write_to_cache(model_json_filename, ret)
    return ret


def _get_model_config(series, model):
    def _get_zhuge():
        start, end = -1, -1
        for i, line in enumerate(model_page.splitlines()):
            if "key: 'viehcle_detail_page_view'," in line: start = i + 1
            if start > 0 and '}' in line:
                end = i + 1
                break
        assert end > start and start > 0, f'{start} {end} wrong with finding zhuge'
        lines = ['{']
        for i, line in enumerate(model_page.splitlines()[start:end]):
            if i == 0:
                line = line.replace('data', '"data"')
            if 'page_title' in line:
                continue
            if i != 0 and len(line.split(':')) == 2:
                k, v = line.split(':')
                v = '"' + v[1:-2].replace('"', "'") + '"'
                line = f'{k}:{v},'
            if i == end - start - 2:
                line = line.rstrip(', ')
            lines.append(line)
        lines.append('}')
        try:
            return json.loads(''.join(lines).replace('\n', '').replace('\t', ''))['data']
        except Exception as e:
            d = ''.join(lines).replace('\n', '')
            print(d.encode('utf-8').decode('utf-8', 'ignore'))
            raise e

    def _get_config():
        start, end = -1, -1
        for i, line in enumerate(model_page.splitlines()):
            if 'id="Table1"' in line: start = i
            if start > 0 and '</table>' in line:
                end = i
                break
        assert end > start and start > 0, f'{start} {end} wrong with finding config'
        content = ''.join(model_page.splitlines()[start:end]).replace('\n', '')
        tds = re.findall('(<td.+?</td>)', content)
        assert len(tds) % 2 == 0, f'{len(tds)} {tds} number is wrong'

        tds = [re.findall('<td [^>]+>(.*)</td>', i)[0].strip() for i in tds]
        for i, td in enumerate(tds):
            r = re.search('>([^<]+)<', td)
            if r:
                td = r.group(1)
            td = td.replace('：', '').strip()
            tds[i] = td
        return tds

    def _get_model_name():
        names = re.findall(f'"/{series}/".+"/m{model}/">([^<]+)', model_page)
        assert len(names) == 1
        return names[0].strip()

    config_json_filename = f'c_{model}.json'
    ret = _read_from_cache(config_json_filename)
    if not ret:
        model_page = _fetch_page(f'{_domain}/m{model}/config.htm', f'c_{model}.html')

        tds = _get_config()

        ret = {tds[i]: tds[i + 1] for i in range(0, len(tds), 2)}
        ret['zhuge'] = _get_zhuge()
        ret['name'] = _get_model_name()
        try:
            print(f'{" "*12}{ret["zhuge"]["model"]}')
        except Exception as e:
            print(ret["zhuge"])
            raise e

        _write_to_cache(config_json_filename, ret)
    return ret


brands = _get_all_brands()
print(f'found {len(brands)} brands')
for brand in brands:
    series = _get_all_series_by(brand)
    print(f'{" "*4}found {len(series)} series for brand {brand["name"]}')
    for serie in series:
        models = _get_all_models_by(serie)
        print(f'{" "*8}found {len(models)} models for series {serie["name"]}')
        for model in models:
            _get_model_config(serie['id'], model)
