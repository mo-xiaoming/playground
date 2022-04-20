#!/usr/bin/env python3

import pika
from datetime import datetime


def callback(ch, method, properties, body):
    print(f' [x] {datetime.now().isoformat()} received "{body.decode()}"')


with pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost')) as connection:
    channel = connection.channel()
    channel.queue_declare(queue='hello')
    channel.basic_consume(
            queue='hello', on_message_callback=callback, auto_ack=True)
    print(' [*] Waiting for messages. To exit press CTRL-C')
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print('stop!')
