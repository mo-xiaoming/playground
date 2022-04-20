#!/usr/bin/env python3

import sys
import pika
import time
import pyperclip
from datetime import datetime


_last_content=pyperclip.paste()

with pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost')) as connection:
    channel = connection.channel()

    channel.queue_declare(queue='hello')

    try:
        while True:
            payload = datetime.now().isoformat()
            channel.basic_publish(exchange='', routing_key='hello', body=payload)

            print(f' [x] {datetime.now().isoformat()} sends "{payload}"')

            time.sleep(.5)
    except KeyboardInterrupt:
        print('stop!')
