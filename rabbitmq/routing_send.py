#!/usr/bin/env python3

import sys
import pika
import time
from datetime import datetime


with pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost')) as connection:
    channel = connection.channel()

    channel.exchange_declare(exchange='direct_logs', exchange_type='direct')

    try:
        while True:
            for severity in ('info', 'warning', 'error'):
                payload = datetime.now().isoformat() + ' ' + severity
                channel.basic_publish(exchange='direct_logs', routing_key=severity, body=payload)
                print(f' [x] {datetime.now().isoformat()} sends "{payload}"')
                time.sleep(.5)
    except KeyboardInterrupt:
        print('Done!')
