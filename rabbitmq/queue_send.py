#!/usr/bin/env python3

import sys
import pika
import time
from datetime import datetime


with pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost')) as connection:
    channel = connection.channel()

    channel.queue_declare(queue='hello', durable=True)

    payload = ' '.join(sys.argv[1:]) or 'Hello World!'
    channel.basic_publish(
            exchange='',
            routing_key='hello',
            body=payload,
            properties=pika.BasicProperties(
                delivery_mode=2
            ))

    print(f' [x] {datetime.now().isoformat()} sends "{payload}"')
