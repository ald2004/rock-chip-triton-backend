import argparse
import numpy as np

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    FLAGS = parser.parse_args()

    # For the HTTP client, need to specify large enough concurrency to
    # issue all the inference requests to the server in parallel. For
    # this example we want to be able to send 2 requests concurrently.
    try:
        concurrent_request_count = 2
        triton_client = httpclient.InferenceServerClient(
            url=FLAGS.url, concurrency=concurrent_request_count)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    # First send a single request to the nonbatching model.
    # print('=========')
    # input0_data = np.array([ 1, 2, 3, 4 ], dtype=np.int32)
    # print('Sending request to nonbatching model: IN0 = {}'.format(input0_data))

    # inputs = [ httpclient.InferInput('IN0', [4], "INT32") ]
    # inputs[0].set_data_from_numpy(input0_data)
    # result = triton_client.infer('nonbatching', inputs)

    # print('Response: {}'.format(result.get_response()))
    # print('OUT0 = {}'.format(result.as_numpy('OUT0')))

    # Send 2 requests to the batching model. Because these are sent
    # asynchronously and Triton's dynamic batcher is configured to
    # delay up to 5 seconds when forming a batch for this model, we
    # expect these 2 requests to be batched within Triton and sent to
    # the minimal backend as a single batch.
    
    async_requests = []

    for _ in range(1):
        print('.',end='',flush=True)
        # input0_data = np.array([[ 10, 11, 12, 13 ]], dtype=np.int32)
        input0_data = np.random.randint(0,high=128,size=(1,3,384,640),dtype=np.int8)
        # print('Sending request to rockchip model: IN0 = {}'.format(input0_data))
        inputs = [ httpclient.InferInput('images', [1,3, 384, 640], "INT8") ]
        inputs[0].set_data_from_numpy(input0_data)
        async_requests.append(triton_client.async_infer('rockchip', inputs))

    # # input0_data = np.array([[ 20, 21, 22, 23 ]], dtype=np.int32)
    # input0_data = np.random.randint(0,high=128,size=(1,3,384,640),dtype=np.int8)
    # print('Sending request to rockchip model: IN0 = {}'.format(input0_data))
    # inputs = [ httpclient.InferInput('INPUT_0', [1,3, 384, 640], "INT8") ]
    # inputs[0].set_data_from_numpy(input0_data)
    
    async_requests.append(triton_client.async_infer('rockchip', inputs))

    for async_request in async_requests:
        # Get the result from the initiated asynchronous inference
        # request. This call will block till the server responds.
        result = async_request.get_result()
        print('Response: {}'.format(result.get_response()))
        print('output = {}'.format(result.as_numpy('output').shape))
        print('371 = {}'.format(result.as_numpy('376').shape))
        print('390 = {}'.format(result.as_numpy('377').shape))
