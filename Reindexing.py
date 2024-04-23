import copy
import numpy as np


def hexagonal_to_rhombohedral_unit_cell(hexagonal_unit_cell, radians):
    a_hexagonal = hexagonal_unit_cell[0]
    c_hexagonal = hexagonal_unit_cell[2]
    a_rhombohedral = 1/3 * np.sqrt(3*a_hexagonal**2 + c_hexagonal**2)
    denom = 2 * np.sqrt(3 + (c_hexagonal/a_hexagonal)**2)
    alpha = 2 * np.arcsin(3 / denom)
    if radians == False:
        alpha *= 180/np.pi
    rhombohedral_unit_cell = np.array([
        a_rhombohedral,
        a_rhombohedral,
        a_rhombohedral,
        alpha,
        alpha,
        alpha,
        ])

    hkl_reindexer = 1/3 * np.array([
        [-1, 2, -1],
        [1, 1, -2],
        [1, 1, 1],
        ])
    return rhombohedral_unit_cell, hkl_reindexer


def hexagonal_to_rhombohedral_hkl(hkl_hexagonal):
    RM = 1/3 * np.array([
        [2, 1, 1],
        [-1, 1, 1],
        [-1, -2, 1],
        ],
        dtype=int
        )
    hkl_rhombohedral = np.matmul(RM, hkl_hexagonal.T).T
    # If you convert to an int without the round, round off error will cause a bug.
    # If hkl == [1.0, 0.99999..., 1.0]
    # conversion to int gives [1, 0, 1]
    hkl_rhombohedral = hkl_rhombohedral.round(decimals=0).astype(int)
    return hkl_rhombohedral


def get_permutation(unit_cell):
    order = np.argsort(unit_cell[:3])
    if np.all(order == [0, 1, 2]):
        permutation = 'abc'
    elif np.all(order == [0, 2, 1]):
        permutation = 'acb'
    elif np.all(order == [1, 0, 2]):
        permutation = 'bac'
    elif np.all(order == [2, 0, 1]):
        permutation = 'cab'
    elif np.all(order == [1, 2, 0]):
        permutation = 'bca'
    elif np.all(order == [2, 1, 0]):
        permutation = 'cba'
    else:
        print(f'Valid permutation could not be determined from {unit_cell}')
        assert False
    permuter = get_permuter(permutation)
    return permutation, permuter


def get_permuter(permutation):
    if permutation in ['abc', 'abc<', 'abc>']:
        permuter = np.eye(3)
    elif permutation in ['acb', 'acb<', 'acb>']:
        # Rx
        permuter = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
            ])
    elif permutation in ['bac', 'bac<', 'bac>']:
        # Rz
        permuter = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
            ])
    elif permutation in ['bca', 'bca<', 'bca>']:
        # Rz Rx
        permuter = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            ])
    elif permutation in ['cab', 'cab<', 'cab>']:
        # Rx Rz
        permuter = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0],
            ])
    elif permutation in ['cba', 'cba<', 'cba>']:
        # Ry
        permuter = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0],
            ])
    else:
        print(f'Permuter could not be determined from {permutation}')
        assert False

    if permutation in ['acb<', 'bac<', 'bca<', 'cab<']:
        # case where angle is acute and at alpha or gamma after reindexing
        permuter = np.matmul(
            permuter,
            np.array([
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, -1],
                ])
            )
    elif permutation in ['abc<', 'cba<']:
        # case where angle is acute and at beta after reindexing
        permuter = np.matmul(
            permuter,
            np.array([
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1],
                ])
            )
    return permuter


def reindex_entry_orthorhombic(unit_cell, spacegroup_symbol, spacegroup_number):
    permutation, permuter = get_permutation(unit_cell)
    permuted_unit_cell = np.concatenate((np.matmul(unit_cell[:3].T, np.abs(permuter)), unit_cell[3:]))
    #           'abc'       'acb',     'bac'       'bca'    'cab',       'cba'
    spacegroup_map_table = {
        16: 'P 2 2 2',
        17: ['P 21 2 2', 'P 21 2 2', 'P 2 21 2', 'P 2 2 21', 'P 2 21 2', 'P 2 2 21'],
        18: ['P 2 21 21', 'P 2 21 21', 'P 21 2 21', 'P 21 21 2', 'P 21 2 21', 'P 21 21 2'],
        19: 'P 21 21 21',
        20: ['C 2 2 21', 'B 2 21 2', 'C 2 2 21', 'B 2 21 2', 'A 21 2 2', 'A 21 2 2'],
        21: ['C 2 2 2', 'B 2 2 2', 'C 2 2 2', 'B 2 2 2', 'A 2 2 2', 'A 2 2 2'],
        22: 'F 2 2 2',
        23: 'I 2 2 2',
        24: 'I 21 21 21',
        25: ['P m m 2', 'P m 2 m', 'P m m 2', 'P m 2 m', 'P 2 m m', 'P 2 m m'],
        26: ['P 21 m a', 'P 21 a m', 'P m 21 b', 'P m c 21', 'P b 21 m', 'P c m 21'],
        27: ['P 2 a a', 'P 2 a a', 'P b 2 b', 'P c c 2', 'P b 2 b', 'P c c 2'],
        28: ['P m 2 a', 'P m a 2', 'P 2 m b', 'P 2 c m', 'P b m 2', 'P c 2 m'],
        29: ['P b c 21', 'P c 21 b', 'P 21 c a', 'P b 21 a', 'P c a 21', 'P 21 a b'],
        30: ['P n c 2', 'P n 2 b', 'P c n 2', 'P b 2 n', 'P 2 n a', 'P 2 a n'],
        31: ['P n m 21', 'P n 21 m', 'P m n 21', 'P m 21 n', 'P 21 n m', 'P 21 m n'],
        32: ['P 2 c b', 'P 2 c b', 'P c 2 a', 'P b a 2', 'P c 2 a', 'P b a 2'],
        33: ['P n a 21', 'P n 21 a', 'P b n 21', 'P c 21 n', 'P 21 n b', 'P 21 c n'],
        34: ['P 2 n n', 'P 2 n n', 'P n 2 n', 'P n n 2', 'P n 2 n', 'P n n 2'],
        35: ['C m m 2', 'B m 2 m', 'C m m 2', 'B m 2 m', 'A 2 m m', 'A 2 m m'],
        36: ['C m c 21', 'B m 21 b', 'C c m 21', 'B b 21 m', 'A 21 m a', 'A 21 a m'],
        37: ['C c c 2', 'B b 2 b', 'C c c 2', 'B b 2 b', 'A 2 a a', 'A 2 a a'],
        38: ['C m 2 m', 'B m m 2', 'C 2 m m', 'B 2 m m', 'A m m 2', 'A m 2 m'],
        39: ['C m 2 a', 'B m a 2', 'C 2 m b', 'B 2 c m', 'A b m 2', 'A c 2 m'],
        40: ['C 2 c m', 'B 2 m b', 'C c 2 m', 'B b m 2', 'A m 2 a', 'A m a 2'],
        41: ['C c 2 a', 'B b a 2', 'C 2 c b', 'B 2 c b', 'A b a 2', 'A c 2 a'],
        42: 'F m m 2',
        43: ['F 2 d d', 'F 2 d d', 'F d 2 d',  'F d d 2', 'F d 2 d', 'F d d 2'],
        44: 'I m m 2',
        45: ['I b a 2', 'I c 2 a', 'I b a 2', 'I c 2 a', 'I 2 c b', 'I 2 c b'],
        46: ['I m a 2', 'I m 2 a', 'I b m 2', 'I c 2 m', 'I 2 m b', 'I 2 c m'],
        47: 'P m m m',
        48: 'P n n n',
        49: ['P m a a', 'P m a a', 'P b m b', 'P c c m', 'P b m b', 'P c c m'],
        50: ['P n c b', 'P n c b', 'P c n a', 'P b a n', 'P c n a', 'P b a n'],
        51: ['P m m a', 'P m a m', 'P m m b', 'P m c m', 'P b m m', 'P c m m'],
        52: ['P n n a', 'P n a n', 'P n n b', 'P n c n', 'P b n n', 'P c n n'],
        53: ['P n c m', 'P n m b', 'P c n m', 'P b m n', 'P m n a', 'P m a n'],
        54: ['P c c b', 'P b c b', 'P c c a', 'P b a a', 'P c a a', 'P b a b'],
        55: ['P m c b', 'P m c b', 'P c m a', 'P b a m', 'P c m a', 'P b a m'],
        56: ['P c c n', 'P b n b', 'P c c n', 'P b n b', 'P n a a', 'P n a a'],
        57: ['P b c m', 'P c m b', 'P m c a', 'P b m a', 'P c a m', 'P m a b'],
        58: ['P m n n', 'P m n n', 'P n m n', 'P n n m', 'P n m n', 'P n n m'],
        59: ['P n m m', 'P n m m', 'P m n m', 'P m m n', 'P m n m', 'P m m n'],
        60: ['P b c n', 'P c n b', 'P c a n', 'P b n a', 'P n c a', 'P n a b'],
        61: ['P b c a', 'P c a b', 'P c a b', 'P b c a', 'P b c a', 'P c a b'],
        62: ['P n a m', 'P n m a', 'P b n m', 'P c m n', 'P m n b', 'P m c n'],
        63: ['C m c m', 'B m m b', 'C c m m', 'B b m m', 'A m m a', 'A m a m'],
        64: ['C c m b', 'B b c m', 'C m c a', 'B m a b', 'A c a m', 'A b m a'],
        65: ['C m m m', 'B m m m', 'C m m m', 'B m m m', 'A m m m', 'A m m m'],
        66: ['C c c m', 'B b m b', 'C c c m', 'B b m b', 'A m a a', 'A m a a'],
        67: ['C m m a', 'B m c m', 'C m m b', 'B m a m', 'A b m m', 'A c m m'],
        68: ['C c c b', 'B b c b', 'C c c a', 'B b a b', 'A b a a', 'A c a a'],
        69: 'F m m m',
        70: 'F d d d',
        71: 'I m m m',
        72: ['I b a m', 'I c m a', 'I b a m', 'I c m a', 'I m c b', 'I m c b'],
        73: 'I b c a',
        74: ['I m c m', 'I m m b', 'I c m m', 'I b m m', 'I m m a', 'I m a m'],
        }
    #           'abc'       'acb',     'bac'       'bca'    'cab',       'cba'
    permuted_spacegroup_symbol = map_spacegroup_symbol(
        spacegroup_map_table, spacegroup_number, spacegroup_symbol, permutation
        )
    return permuted_spacegroup_symbol, permutation, permuted_unit_cell, permuter


def map_spacegroup_symbol(spacegroup_map_table, key, spacegroup_symbol, permutation):
    if isinstance(spacegroup_map_table[key], str):
        permuted_spacegroup_symbol = spacegroup_symbol
    elif isinstance(spacegroup_map_table[key], list):
        current_index = spacegroup_map_table[key].index(spacegroup_symbol)
        if current_index == 0:
            # abc
            if permutation in ['abc', 'abc<', 'abc>']:
                new_index = 0
            elif permutation in ['acb', 'acb<', 'acb>']:
                new_index = 1
            elif permutation in ['bac', 'bac<', 'bac>']:
                new_index = 2
            elif permutation in ['bca', 'bca<', 'bca>']:
                new_index = 3
            elif permutation in ['cab', 'cab<', 'cab>']:
                new_index = 4
            elif permutation in ['cba', 'cba<', 'cba>']:
                new_index = 5
        elif current_index == 1:
            # acb
            if permutation in ['abc', 'abc<', 'abc>']:
                new_index = 1
            elif permutation in ['acb', 'acb<', 'acb>']:
                new_index = 0
            elif permutation in ['bac', 'bac<', 'bac>']:
                new_index = 4
            elif permutation in ['bca', 'bca<', 'bca>']:
                new_index = 5
            elif permutation in ['cab', 'cab<', 'cab>']:
                new_index = 2
            elif permutation in ['cba', 'cba<', 'cba>']:
                new_index = 3
        elif current_index == 2:
            # bac
            if permutation in ['abc', 'abc<', 'abc>']:
                new_index = 2
            elif permutation in ['acb', 'acb<', 'acb>']:
                new_index = 3
            elif permutation in ['bac', 'bac<', 'bac>']:
                new_index = 0
            elif permutation in ['bca', 'bca<', 'bca>']:
                new_index = 1
            elif permutation in ['cab', 'cab<', 'cab>']:
                new_index = 5
            elif permutation in ['cba', 'cba<', 'cba>']:
                new_index = 4
        elif current_index == 3:
            # bca
            if permutation in ['abc', 'abc<', 'abc>']:
                new_index = 3
            elif permutation in ['acb', 'acb<', 'acb>']:
                new_index = 2
            elif permutation in ['bac', 'bac<', 'bac>']:
                new_index = 5
            elif permutation in ['bca', 'bca<', 'bca>']:
                new_index = 4
            elif permutation in ['cab', 'cab<', 'cab>']:
                new_index = 0
            elif permutation in ['cba', 'cba<', 'cba>']:
                new_index = 1
        elif current_index == 4:
            # cab
            if permutation in ['abc', 'abc<', 'abc>']:
                new_index = 4
            elif permutation in ['acb', 'acb<', 'acb>']:
                new_index = 5
            elif permutation in ['bac', 'bac<', 'bac>']:
                new_index = 1
            elif permutation in ['bca', 'bca<', 'bca>']:
                new_index = 0
            elif permutation in ['cab', 'cab<', 'cab>']:
                new_index = 3
            elif permutation in ['cba', 'cba<', 'cba>']:
                new_index = 2
        elif current_index == 5:
            # cba
            if permutation in ['abc', 'abc<', 'abc>']:
                new_index = 5
            elif permutation in ['acb', 'acb<', 'acb>']:
                new_index = 4
            elif permutation in ['bac', 'bac<', 'bac>']:
                new_index = 3
            elif permutation in ['bca', 'bca<', 'bca>']:
                new_index = 2
            elif permutation in ['cab', 'cab<', 'cab>']:
                new_index = 1
            elif permutation in ['cba', 'cba<', 'cba>']:
                new_index = 0
        permuted_spacegroup_symbol = spacegroup_map_table[key][new_index]
    return permuted_spacegroup_symbol


def reindex_entry_monoclinic(unit_cell, spacegroup_symbol, radians):
    # Useful resources:
    #    http://pd.chem.ucl.ac.uk/pdnn/symm4/practice.htm
    #    https://onlinelibrary.wiley.com/iucr/itc/Ab/ch5o1v0001/sec5o1o3.pdf
    if radians:
        check = np.pi/2
        angle_multiplier = 1
    else:
        check = 90
        angle_multiplier = np.pi/180

    A_centered = ['A 1 2 1', 'A 1 m 1', 'A 1 a 1', 'A 1 2/m 1', 'A 1 2/a 1']
    B_centered = [
        'B 1 2 1', 'B 1 21 1', 'B 1 m 1', 'B 1 a 1', 'B 1 d 1', 'B 1 2/m 1',
        'B 1 21/m 1', 'B 1 2/a 1', 'B 1 2/d 1', 'B 1 21/a 1'
        ]
    I_centered = ['I 1 2 1', 'I 1 m 1', 'I 1 a 1', 'I 1 2/m 1', 'I 1 2/a 1']
    standard_settings = [
        'P 1 2 1', 'P 1 21 1', 'C 1 2 1', 'P 1 m 1', 'P 1 c 1',
        'C 1 m 1', 'C 1 c 1', ' P 1 2/m 1', 'P 1 21/m 1', 'C 1 2/m 1',
        'P 1 2/c 1', 'P 1 21/c 1', 'C 1 2/c 1'
        ]
    ac_equivalent = ['P 1 2 1', 'P 1 m 1', 'P 1 2/m 1', 'P 1 21 1', 'P 1 21/m 1']
    Pa_symbols = ['P 1 a 1', 'P 1 2/a 1', 'P 1 21/a 1']
    Pn_symbols = ['P 1 n 1', 'P 1 2/n 1', 'P 1 21/n 1']

    reindexed_spacegroup_symbol = copy.copy(spacegroup_symbol)

    a = unit_cell[0]
    b = unit_cell[1]
    c = unit_cell[2]
    beta = angle_multiplier * unit_cell[4]
    ucm = np.array([
        [a, 0, c*np.cos(beta)],
        [0, b, 0],
        [0, 0, c*np.sin(beta)],
        ])
    if spacegroup_symbol in standard_settings:
        centered_reindexer = np.eye(3)
    elif spacegroup_symbol in A_centered + Pa_symbols:
        # A centered settings are converted to C centered
        # primitive settings with a glide plane along the a axis are convereted to the setting
        #   with the glide plane along the c axis
        centered_reindexer = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0],
            ])
        if spacegroup_symbol in A_centered:
            reindexed_spacegroup_symbol = reindexed_spacegroup_symbol.replace('A', 'C')
        reindexed_spacegroup_symbol = reindexed_spacegroup_symbol.replace('a', 'c')
    elif spacegroup_symbol in I_centered:
        # I centered settings are converted to C centered
        centered_reindexer = np.array([
            [-1, 0, 1],
            [0, 1, 0],
            [-1, 0, 0],
            ])
        reindexed_spacegroup_symbol = reindexed_spacegroup_symbol.replace('I', 'C')
        reindexed_spacegroup_symbol = reindexed_spacegroup_symbol.replace('a', 'c')
    elif spacegroup_symbol in Pn_symbols:
        # primitive settings with a glide plane along the n axis are convereted to the setting
        #   with the glide plane along the c axis
        centered_reindexer = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 1],
            ])
        reindexed_spacegroup_symbol = reindexed_spacegroup_symbol.replace('n', 'c')
    elif spacegroup_symbol in ['I 1 2/c 1', 'I 1 c 1']:
        centered_reindexer = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [-1, 0, 1],
            ])
        reindexed_spacegroup_symbol = reindexed_spacegroup_symbol.replace('I', 'C')
    elif spacegroup_symbol == ['A 1 n 1', 'A 1 2/n 1']:
        centered_reindexer = np.array([
            [0, 0, -1],
            [0, 1, 0],
            [1, 0, -1],
            ])
        reindexed_spacegroup_symbol = reindexed_spacegroup_symbol.replace('A', 'C')
        reindexed_spacegroup_symbol = reindexed_spacegroup_symbol.replace('n', 'c')
    elif spacegroup_symbol in ['C 1 n 1', 'C 1 2/n 1']:
        centered_reindexer = np.array([
            [-1, 0, 1],
            [0, -1, 0],
            [0, 0, 1],
            ])
        reindexed_spacegroup_symbol = reindexed_spacegroup_symbol.replace('n', 'c')
    else:
        # These cases entries that are reported in
        # unexpected, nonstandard settings:
        # And in settings so the b/y axis is not unique:
        #   'P 1 1 21/n', 'P 1 1 21' 'I 2/b 1 1', 'P 21/c 1 1', 'C 2/m 1 1', 'P 21/m 1 1'
        return None, None, np.eye(3)

    if spacegroup_symbol in ac_equivalent and a > c:
        # primitive settings with no symmetry elements along the a or c axis are reindexed so
        #   a < c
        ac_reindexer = np.array([
                [0, 0, 1],
                [0, 1, 0],
                [-1, 0, 0],
                ])
        centered_reindexer = ac_reindexer @ centered_reindexer
    
    rucm = ucm @ centered_reindexer
    reindexed_unit_cell = np.zeros(6)
    reindexed_unit_cell[0] = np.linalg.norm(rucm[:, 0])
    reindexed_unit_cell[1] = np.linalg.norm(rucm[:, 1])
    reindexed_unit_cell[2] = np.linalg.norm(rucm[:, 2])
    dot_product = np.dot(rucm[:, 0], rucm[:, 2])
    mag = reindexed_unit_cell[0] * reindexed_unit_cell[2]
    reindexed_unit_cell[4] = np.arccos(dot_product / mag) / angle_multiplier
    reindexed_unit_cell[3] = check
    reindexed_unit_cell[5] = check
    
    if reindexed_unit_cell[4] < check:
        obtuse_reindexer = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
            ])
        reindexed_unit_cell[4] = 2*check - reindexed_unit_cell[4]
        hkl_reindexer = obtuse_reindexer @ centered_reindexer
    else:
        hkl_reindexer = centered_reindexer
        
    return reindexed_unit_cell, reindexed_spacegroup_symbol, hkl_reindexer
