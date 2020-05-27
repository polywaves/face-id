import config
import grpc
from google.protobuf.wrappers_pb2 import BoolValue, Int32Value
import cameras_pb2
import cameras_pb2_grpc


def get_camera():
    server = str(config.get('HOST')) + ':' + str(config.get('GRPC_PORT'))

    with grpc.insecure_channel(server) as channel:
        stub = cameras_pb2_grpc.CamerasStub(channel)
        filter_by_activity = BoolValue()
        filter_by_activity.value = True
        response = stub.getCamerasList(cameras_pb2.ListChallenge(is_active=filter_by_activity))

    result = None
    for camera in response.cameras:
        if int(camera.id) == int(config.get('CAMERA_ID')):
            result = camera

    return result
