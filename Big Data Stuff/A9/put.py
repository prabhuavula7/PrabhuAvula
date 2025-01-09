from kafka import KafkaProducer

bootstrap_servers = ['localhost:9092']  

producer = KafkaProducer(bootstrap_servers=bootstrap_servers)

messages = [
    (b'MYID', b'A20522815'),
    (b'MYNAME', b'Prabhu Avula'),
    (b'MYEYECOLOR', b'brown')  
]

for key, value in messages:
    producer.send('sample', key=key, value=value)

producer.flush()
producer.close()

print("Messages sent successfully.")
