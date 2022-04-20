#!/usr/bin/env python3

import pika
from datetime import datetime
import time


def callback(ch, method, properties, body):
    print(f' [x] {datetime.now().isoformat()} received "{body.decode()}"')
    time.sleep(body.count(b'.'))
    print(f' [x] {datetime.now().isoformat()} Done')
    ch.basic_ack(delivery_tag = method.delivery_tag)


with pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost')) as connection:
    channel = connection.channel()
    channel.queue_declare(queue='hello', durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='hello', on_message_callback=callback)
    print(' [*] Waiting for messages. To exit press CTRL-C')
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print('stop!')
