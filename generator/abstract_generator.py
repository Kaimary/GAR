import abc

class AbstractSQLGenerator(metaclass=abc.ABCMeta):
    '''Used for generating sampled SQL queries
    '''
    def __init__(self, tables_file, db_dir):
        """
        :param tables_file: database schema file
        :param db_dir: The directory of databases for the dataset.
        """
        self.tables_file = tables_file
        self.db_dir = db_dir

    @abc.abstractmethod
    def generate(self):
        """
        Generate sqls based on the current database and context.

        :return: a list of sqls.
        """
        pass
    
    @abc.abstractmethod
    def switch_context(self, *args):
        """
        Switch to the current specific context for generation
        """
        pass

    @abc.abstractmethod
    def switch_database(self, db_name):
        """
        Switch to the current underlying database and load the schema

        :param db_name: The name of database that needs to do the generation.
        """
        pass