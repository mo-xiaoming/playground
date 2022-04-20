#!/usr/bin/env python3

import sys
import pika
import time
from datetime import datetime


with pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost')) as connection:
    channel = connection.channel()

    channel.exchange_declare(exchange='logs', exchange_type='fanout')

    try:
        while True:
            payload = datetime.now().isoformat()
            channel.basic_publish(exchange='logs', routing_key='', body=payload)
            print(f' [x] {datetime.now().isoformat()} sends "{payload}"')
            time.sleep(.5)
    except KeyboardInterrupt:
        print('Done!')
