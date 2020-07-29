# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import cameras_pb2 as cameras__pb2


class CamerasStub(object):
    """Service to work with camera entities
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.getCamerasList = channel.unary_unary(
                '/cameras.Cameras/getCamerasList',
                request_serializer=cameras__pb2.ListChallenge.SerializeToString,
                response_deserializer=cameras__pb2.ListReply.FromString,
                )


class CamerasServicer(object):
    """Service to work with camera entities
    """

    def getCamerasList(self, request, context):
        """method to request list of cameras
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_CamerasServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'getCamerasList': grpc.unary_unary_rpc_method_handler(
                    servicer.getCamerasList,
                    request_deserializer=cameras__pb2.ListChallenge.FromString,
                    response_serializer=cameras__pb2.ListReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'cameras.Cameras', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Cameras(object):
    """Service to work with camera entities
    """

    @staticmethod
    def getCamerasList(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/cameras.Cameras/getCamerasList',
            cameras__pb2.ListChallenge.SerializeToString,
            cameras__pb2.ListReply.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)
