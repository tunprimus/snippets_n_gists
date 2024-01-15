#Faster Python APIs in Two Minutes
#https://github.com/paul-armstrong-dev/faster-parser
import concurrent.futures
import pandas as pd
from loguru import logger


def fast_parse(python_class, parse_function, data_to_parse, number_of_workers = 4, **kwargs):
    """
    Util function to split any data set to the number of workers, then return results using any given parsing function.

    NOte that when using dicts, the index of the key will be passed to the function object too, so that needs to be handled.
        :param python_class: Instantiated class object which contains the parse function
        :param parse_function: Function to parse data, can either be a list or a dict
        :param data_to_parse: Data to be parsed.
        :param number_of_workers: Number of workers to split the parsing to.
        :param kwargs: Optional, extra params which parse function may need.
        :return:
    """
    try:
        function_object = getattr(python_class, parse_function)
    except AttributeError as e:
        logger.error(f"{python_class} does not have {parse_function}")
        return
    else:
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers = number_of_workers) as executor:
            if type(data_to_parse) == list:
                future_to_result = {executor.submit(function_object, data, **kwargs): data for data in data_to_parse}
            elif type(data_to_parse) == dict or type(data_to_parse) == pd.Series:
                for index, data in data_to_parse.items():
                    future_to_result = {executor.submit(function_object, index, data, **kwargs)}
            else:
                logger.error("Unsupported data type")
                return
            
            for future in concurrent.futures.as_completed(future_to_result):
                try:
                    data = future.result()
                except Exception as exc:
                    logger.error(f"{future_to_result[future]} generated an exception: {exc}")
                finally:
                    results.append(data)
            return results