
def benchmark(func):
    def wrapper(*args, **kwargs):
        from time import time

        time_start = time()
        result = func(*args, **kwargs)
        time_finish = time()

        time_delta = time_finish - time_start
        print('- [{func_name}] fonksiyonu {seconds} saniye sürdü.'.format(
            func_name=func.__name__,
            seconds=round(time_delta, 2),
        ))

        return result

    return wrapper




@benchmark
def test_response_status_is_200():
    from urllib import request

    response = request.urlopen(
        request.Request('https://httpbin.org/status/200',
                        method='HEAD')
    )

    assert response.status == 200


test_response_status_is_200()