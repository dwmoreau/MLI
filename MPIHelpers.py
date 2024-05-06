import logging
from mpi4py import MPI


def get_mpi_logger(rank, save_to, tag):
    logger = logging.getLogger("rank[%i]" % rank)
    logger.setLevel(logging.DEBUG)
    mh = MPIFileHandler(f'{save_to}/{tag}.log')
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
    mh.setFormatter(formatter)
    logger.addHandler(mh)
    return logger


class MPIFileHandler(logging.FileHandler):
    """
    Code was copied from https://gist.github.com/muammar/2baec60fa8c7e62978720686895cdb9f

    Created on Wed Feb 14 16:17:38 2018
    This handler is used to deal with logging with mpi4py in Python3.
    @author: cheng
    @reference:
        https://cvw.cac.cornell.edu/python/logging
        https://groups.google.com/forum/#!topic/mpi4py/SaNzc8bdj6U
        https://gist.github.com/JohnCEarls/8172807
    """
    def __init__(self,
                 filename,
                 mode=MPI.MODE_WRONLY|MPI.MODE_CREATE|MPI.MODE_APPEND ,
                 encoding='utf-8',
                 delay=False,
                 comm=MPI.COMM_WORLD):
        self.baseFilename = os.path.abspath(filename)
        self.mode = mode
        self.encoding = encoding
        self.comm = comm
        if delay:
            #We don't open the stream, but we still need to call the
            #Handler constructor to set level, formatter, lock etc.
            logging.Handler.__init__(self)
            self.stream = None
        else:
            logging.StreamHandler.__init__(self, self._open())

    def _open(self):
        stream = MPI.File.Open( self.comm, self.baseFilename, self.mode )
        stream.Set_atomicity(True)
        return stream

    def emit(self, record):
        """
        Emit a record.
        If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.  If
        exception information is present, it is formatted using
        traceback.print_exception and appended to the stream.  If the stream
        has an 'encoding' attribute, it is used to determine how to do the
        output to the stream.

        Modification:
            stream is MPI.File, so it must use `Write_shared` method rather
            than `write` method. And `Write_shared` method only accept
            bytestring, so `encode` is used. `Write_shared` should be invoked
            only once in each all of this emit function to keep atomicity.
        """
        try:
            msg = self.format(record)
            stream = self.stream
            stream.Write_shared((msg+self.terminator).encode(self.encoding))
            #self.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        if self.stream:
            self.stream.Sync()
            self.stream.Close()
            self.stream = None
