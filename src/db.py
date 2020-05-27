import os
import config
from peewee import Model, PostgresqlDatabase, IntegerField, TextField, PrimaryKeyField, ForeignKeyField, BooleanField, DateField, BlobField

db = PostgresqlDatabase(
    database=config.get('POSTGRES_DB'),
    user=config.get('POSTGRES_USER'),
    password=config.get('POSTGRES_PASSWORD'),
    host=config.get('HOST'),
    port=config.get('POSTGRES_PORT'),
    autocommit=False,
    autorollback=True
)


class BaseModel(Model):
    class Meta:
        database = db


class ObjectFaces(BaseModel):
    id = IntegerField(primary_key=True)
    data = BlobField()
    object_id = IntegerField(default=0)
    created_at = DateField()
    updated_at = DateField()
    deleted_at = DateField()

    class Meta:
        table_name = 'object_faces'


class Objects(BaseModel):
    id = IntegerField(primary_key=True)
    individual_id = IntegerField(default=0)
    camera_id = IntegerField(default=0)
    created_at = DateField()
    updated_at = DateField()
    deleted_at = DateField()


class RecognitionModels(BaseModel):
    id = IntegerField(primary_key=True)
    data = BlobField()
    created_at = DateField()
    updated_at = DateField()
    deleted_at = DateField()

    class Meta:
        table_name = 'recognition_models'


def init():
    db.connect()
    db.create_tables([ObjectFaces, Objects, RecognitionModels])
