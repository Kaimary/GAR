import abc
from typing import List, Union

class AbstractNLSynthesizer(metaclass=abc.ABCMeta):
    '''Used for synthesizing NL expressions for corresponding SQLs
    '''

    @abc.abstractmethod
    def synthesize(self, sql: Union[str, List[str]], batch=False):
        """
        Synthesize NL expression based on the given SQL

        :return: NL expression string
        """
        pass
    