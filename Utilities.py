import csv
import logging
from mpi4py import MPI
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """
    If you try to do regression on only cubic, tetragonal and orthorhombic crystals, all the angles
    are 90 degrees and excluded from the training the transformer. This is a bit of a hack, where
    that case is caught and the transformer is switch to the IdentityTransformer. The rest of the
    code just runs without any changes.
    Copied from:
        https://gist.github.com/laurentmih/e2efed8d1e0679e88c03460c5a9e2520
    """
    def __init__(self):
        self.scale_ = [1]
        self.mean_ = [0]
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array):
        return input_array

    def inverse_transform(self, input_array):
        return input_array


class Q2Calculator:
    def __init__(self, lattice_system, hkl, tensorflow):
        self.hkl = hkl
        self.lattice_system = lattice_system
        if tensorflow:
            import tensorflow as tf
            self.newaxis = tf.newaxis
            self.sin = tf.math.sin
            self.cos = tf.math.cos
            self.zeros = tf.zeros
            self.zeros_like = tf.zeros_like
        else:
            self.newaxis = np.newaxis
            self.sin = np.sin
            self.cos = np.cos
            self.zeros = np.zeros
            self.zeros_like = np.zeros_like
        if self.lattice_system == 'cubic':
            self.get_q2 = self.get_q2_cubic
        elif self.lattice_system == 'tetragonal':
            self.get_q2 = self.get_q2_tetragonal
        elif self.lattice_system == 'orthorhombic':
            self.get_q2 = self.get_q2_orthorhombic
        elif self.lattice_system == 'monoclinic':
            self.get_q2 = self.get_q2_monoclinic
        elif self.lattice_system == 'triclinic':
            self.get_q2 = self.get_q2_triclinic
        elif self.lattice_system == 'hexagonal':
            self.get_q2 = self.get_q2_hexagonal
        elif self.lattice_system == 'rhombohedral':
            self.get_q2 = self.get_q2_rhombohedral

    def get_q2_cubic(self, unit_cell):
        a = unit_cell[:, 0][:, self.newaxis]
        q2_ref = (self.hkl[:, 0]**2 + self.hkl[:, 1]**2 + self.hkl[:, 2]**2) / a**2
        return q2_ref

    def get_q2_tetragonal(self, unit_cell):
        a = unit_cell[:, 0][:, self.newaxis]
        c = unit_cell[:, 1][:, self.newaxis]
        q2_ref = (self.hkl[:, 0]**2 + self.hkl[:, 1]**2) / a**2 + self.hkl[:, 2]**2 / c**2
        return q2_ref

    def get_q2_orthorhombic(self, unit_cell):
        a = unit_cell[:, 0][:, self.newaxis] 
        b = unit_cell[:, 1][:, self.newaxis]
        c = unit_cell[:, 2][:, self.newaxis]
        q2_ref = self.hkl[:, 0]**2 / a**2 + self.hkl[:, 1]**2 / b**2 + self.hkl[:, 2]**2 / c**2
        return q2_ref

    def get_q2_monoclinic(self, unit_cell):
        a = unit_cell[:, 0][:, self.newaxis] 
        b = unit_cell[:, 1][:, self.newaxis]
        c = unit_cell[:, 2][:, self.newaxis]
        cos_beta = self.cos(unit_cell[:, 3][:, self.newaxis])
        sin_beta = self.sin(unit_cell[:, 3][:, self.newaxis])

        term0 = self.hkl[:, 0]**2 / a**2
        term1 = self.hkl[:, 1]**2 * sin_beta**2 / b**2
        term2 = self.hkl[:, 2]**2 / c**2
        term3 = 2 * self.hkl[:, 0] * self.hkl[:, 2] * cos_beta / (a * c)
        term4 = 1 / sin_beta**2
        q2_ref = term4 * (term0 + term1 + term2 + term3)
        return q2_ref

    def get_q2_triclinic(self, unit_cell):
        a = unit_cell[:, 0][:, self.newaxis]
        b = unit_cell[:, 1][:, self.newaxis]
        c = unit_cell[:, 2][:, self.newaxis]
        calpha = self.cos(unit_cell[:, 3])[:, self.newaxis]
        cbeta = self.cos(unit_cell[:, 4])[:, self.newaxis]
        cgamma = self.cos(unit_cell[:, 5])[:, self.newaxis]
        salpha = self.sin(unit_cell[:, 3])[:, self.newaxis]
        sbeta = self.sin(unit_cell[:, 4])[:, self.newaxis]
        sgamma = self.sin(unit_cell[:, 5])[:, self.newaxis]

        denom = 1 + 2*calpha*cbeta*cgamma - calpha**2 - cbeta**2 - cgamma**2
        term0 = self.hkl[:, 0]**2 * salpha**2 / a**2
        term1 = self.hkl[:, 1]**2 * sbeta**2 / b**2
        term2 = self.hkl[:, 2]**2 * sgamma**2 / c**2
        term3a = 2*self.hkl[:, 0]*self.hkl[:, 1] / (a*b)
        term3b = calpha*cbeta - cgamma
        term3 = term3a * term3b
        term4a = 2*self.hkl[:, 1]*self.hkl[:, 2] / (b*c)
        term4b = cbeta*cgamma - calpha
        term4 = term4a * term4b
        term5a = 2*self.hkl[:, 0]*self.hkl[:, 2] / (a*c)
        term5b = calpha*cgamma - cbeta
        term5 = term5a * term5b
        q2_ref = (term0 + term1 + term2 + term3 + term4 + term5) / denom
        return q2_ref

    def get_q2_rhombohedral(self, unit_cell):
        a = unit_cell[:, 0][:, self.newaxis]
        cos_alpha = self.cos(unit_cell[:, 1][:, self.newaxis])
        sin_alpha = self.sin(unit_cell[:, 1][:, self.newaxis])

        h = self.hkl[:, 0]
        k = self.hkl[:, 1]
        l = self.hkl[:, 2]
        term0 = (h**2 + k**2 + l**2) * sin_alpha**2
        term1 = 2 * (h*k + k*l + l*h)
        term2 = cos_alpha**2 - cos_alpha
        term3 = a**2 * (1 + 2*cos_alpha**3 - 3*cos_alpha**2)
        q2_ref = (term0 + term1 * term2) / term3
        return q2_ref

    def get_q2_hexagonal(self, unit_cell):
        a = unit_cell[:, 0][:, self.newaxis]
        c = unit_cell[:, 1][:, self.newaxis]

        h = self.hkl[:, 0]
        k = self.hkl[:, 1]
        l = self.hkl[:, 2]
        q2_ref = 4/3 * (h**2 + h*k + k**2) / a**2 + l**2 / c**2
        return q2_ref


class PairwiseDifferenceCalculator(Q2Calculator):
    def __init__(self, lattice_system, hkl_ref, tensorflow, q2_scaler, uc_scaler=None, angle_scale=None):
        super().__init__(lattice_system, hkl_ref, tensorflow)
        self.q2_scaler = q2_scaler
        if uc_scaler is not None:
            self.uc_scaler = uc_scaler
            self.angle_scale = angle_scale

    def get_pairwise_differences_from_uc_scaled(self, uc_pred_scaled, q2_scaled):
        uc_pred = self.zeros_like(uc_pred_scaled)
        if self.lattice_system in ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal']:
            uc_pred = uc_pred_scaled * self.uc_scaler.scale_[0] + self.uc_scaler.mean_[0]
        elif self.lattice_system == 'monoclinic':
            uc_pred[:, :3] = uc_pred_scaled[:, :3] * self.uc_scaler.scale_[0] + self.uc_scaler.mean_[0]
            uc_pred[:, 3] = self.angle_scale * uc_pred_scaled[:, 3] + np.pi/2
        elif self.lattice_system == 'triclinic':
            uc_pred[:, :3] = uc_pred_scaled[:, :3] * self.uc_scaler.scale_[0] + self.uc_scaler.mean_[0]
            uc_pred[:, 3:] = self.angle_scale * uc_pred_scaled[:, 3:] + np.pi/2
        elif self.lattice_system == 'rhombohedral':
            uc_pred[:, 0] = uc_pred_scaled[:, 0] * self.uc_scaler.scale_[0] + self.uc_scaler.mean_[0]
            uc_pred[:, 1] = self.angle_scale * uc_pred_scaled[:, 1] + np.pi/2
        return self.get_pairwise_differences(uc_pred, q2_scaled)

    def get_pairwise_differences(self, uc_pred, q2_scaled):
        q2_ref = self.get_q2(uc_pred)
        q2_ref_scaled = (q2_ref - self.q2_scaler.mean_[0]) / self.q2_scaler.scale_[0]
        # d_spacing_ref: n x hkl_ref_length
        # x: n x 10
        # differences = n x 10 x hkl_ref_length
        pairwise_differences_scaled = (q2_ref_scaled[:, self.newaxis, :] - q2_scaled[:, :, self.newaxis]) / np.sqrt(2)
        return pairwise_differences_scaled


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


def write_params(params, filename):
    with open(filename, 'w') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=params.keys())
        writer.writeheader()
        writer.writerow(params)

def read_params(filename):
    with open(filename, 'r') as params_file:
        reader = csv.DictReader(params_file)
        for row in reader:
            params = row
    return params
