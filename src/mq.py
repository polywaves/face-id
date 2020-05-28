import config
import pika


class Mq:
    def __init__(
        self,
        queue='recognition_records_receive',
        exchange='recognition_records',
        routing_key='new_record'
    ):
        self.username = config.get('RABBITMQ_USER')
        self.password = config.get('RABBITMQ_PASSWORD')
        self.host = config.get('HOST')
        self.virtual_host = config.get('RABBITMQ_VIRTUAL_HOST')
        self.port = config.get('RABBITMQ_PORT')
        self.queue = queue
        self.exchange = exchange
        self.routing_key = routing_key

        self.connection = None
        self.channel = None

        connection_name = 'face-id_cam-' + config.get('CAMERA_ID')

        credentials = pika.PlainCredentials(self.username, self.password)
        self.parameters = pika.ConnectionParameters(
            self.host,
            self.port,
            self.virtual_host,
            credentials,
            client_properties={'connection_name': connection_name}
        )

    def connect(self):
        self.connection = pika.BlockingConnection(self.parameters)
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue, durable=True)

    def close(self):
        self.connection.close()

    def get(self):
        self.channel.queue_bind(
            queue=self.queue,
            exchange=self.exchange,
            routing_key=self.routing_key
        )

        try:
            method_frame, header_frame, body = self.channel.basic_get(queue=self.queue)
            self.channel.basic_ack(delivery_tag=method_frame.delivery_tag)

            return body
        except Exception:
            return None

    def consume(self, callback):
        self.channel.queue_bind(
            queue=self.queue,
            exchange=self.exchange,
            routing_key=self.routing_key
        )

        try:
            self.channel.basic_consume(self.queue, callback)
            self.channel.start_consuming()
        except pika.exceptions.AMQPConnectionError:
            self.connect()
            self.consume(callback)

    def send(self, data, routing_key=None):
        _routing_key = self.routing_key
        if routing_key:
            _routing_key = routing_key

        try:
            self.channel.basic_publish(
                exchange=self.exchange,
                routing_key=_routing_key,
                body=data,
                properties=pika.BasicProperties(
                    delivery_mode=2
                )
            )
        except pika.exceptions.AMQPConnectionError:
            self.connect()
            self.send(data)
