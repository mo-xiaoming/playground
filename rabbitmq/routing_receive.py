#!/usr/bin/env python3

import pika
from datetime import datetime
import time
import sys


def callback(ch, method, properties, body):
    print(f' [x] {datetime.now().isoformat()} received "{body.decode()}"')


severities = sys.argv[1:]
if not severities:
    sys.exit(0)


with pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost')) as connection:
    channel = connection.channel()

    channel.exchange_declare(exchange='direct_logs', exchange_type='direct')

    result = channel.queue_declare(queue='', exclusive=True)

    queue_name = result.method.queue
    print(f' [ ] bind to {queue_name}')

    for severity in severities:
        channel.queue_bind(exchange='direct_logs', queue=queue_name, routing_key=severity)

    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL-C')
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print('stop!')
