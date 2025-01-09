from kafka import KafkaConsumer
bootstrap_servers = ['localhost:9092']
topic = 'sample'
consumer = KafkaConsumer(
    topic,
    bootstrap_servers=bootstrap_servers,
    group_id='my-group',
    auto_offset_reset='earliest'
)

try:
    for message in consumer:
        key = message.key.decode('utf-8') if message.key else None
        value = message.value.decode('utf-8') if message.value else None
        print(f"Key={key}, Value={value}")
        break
finally:
    consumer.close()
