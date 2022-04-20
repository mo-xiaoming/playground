#!/usr/bin/env python3

import pika
from datetime import datetime
import time


def callback(ch, method, properties, body):
    print(f' [x] {datetime.now().isoformat()} received "{body.decode()}"')


with pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost')) as connection:
    channel = connection.channel()

    channel.exchange_declare(exchange='logs', exchange_type='fanout')

    result = channel.queue_declare(queue='', exclusive=True)

    queue_name = result.method.queue
    print(f' [ ] bind to {queue_name}')

    channel.queue_bind(exchange='logs', queue=queue_name)

    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL-C')
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print('stop!')
