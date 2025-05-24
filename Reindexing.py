import copy
import numpy as np


def hexagonal_to_rhombohedral_unit_cell(hexagonal_unit_cell):
    a_hexagonal = hexagonal_unit_cell[0]
    c_hexagonal = hexagonal_unit_cell[2]
    a_rhombohedral = 1/3 * np.sqrt(3*a_hexagonal**2 + c_hexagonal**2)
    denom = 2 * np.sqrt(3 + (c_hexagonal/a_hexagonal)**2)
    alpha = 2 * np.arcsin(3 / denom)
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


def reindex_entry_orthorhombic(unit_cell, spacegroup_symbol, spacegroup_number):
    if spacegroup_symbol[0] in ['P', 'I', 'F']:
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
            print(f'Valid permutation could not be determined from {unit_cell} {spacegroup_symbol}')
            assert False
    else:
        if spacegroup_symbol[0] == 'C':
            # Center on AB face. Can swap A & B axes
            if unit_cell[0] <= unit_cell[1]:
                permutation = 'abc'
            else:
                permutation = 'bac'
        elif spacegroup_symbol[0] == 'B':
            # Center on AC face. Must swap B and C axes 'acb'
            # Can swap A & C axes
            if unit_cell[0] <= unit_cell[2]:
                permutation = 'acb'
            else:
                permutation = 'cab'
        elif spacegroup_symbol[0] == 'A':
            # Center on BC face. Must swap A and C axes 'cba'
            # Can swap B & C axes
            if unit_cell[2] <= unit_cell[1]:
                permutation = 'cba'
            else:
                permutation = 'bca'
        else:
            print(f'Valid permutation could not be determined from {unit_cell} {spacegroup_symbol}')
            assert False

    if permutation == 'abc':
        permuter = np.eye(3)
    elif permutation == 'acb':
        # Rx
        permuter = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
            ])
    elif permutation == 'bac':
        # Rz
        permuter = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
            ])
    elif permutation == 'bca':
        # Rz Rx
        permuter = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            ])
    elif permutation == 'cab':
        # Rx Rz
        permuter = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0],
            ])
    elif permutation == 'cba':
        # Ry
        permuter = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0],
            ])
    else:
        print(f'Permuter could not be determined from {permutation}')
        assert False

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


def get_split_group(lattice_system, unit_cell=None, reciprocal_reindexed_unit_cell=None, reindexed_spacegroup_symbol_hm=None):
    if lattice_system in ['cubic', 'rhombohedral', 'triclinic']:
        split = 0
    elif lattice_system in ['tetragonal', 'hexagonal']:
        if unit_cell[0] < unit_cell[2]:
            split = 0
        else:
            split = 1
    elif lattice_system == 'monoclinic':
        # splitting based on reciprocal space
        if reciprocal_reindexed_unit_cell[0] >= reciprocal_reindexed_unit_cell[1]:
            # a* > b*
            if reciprocal_reindexed_unit_cell[1] >= reciprocal_reindexed_unit_cell[2]:
                split = 0 # cba* (abc)
            elif reciprocal_reindexed_unit_cell[0] >= reciprocal_reindexed_unit_cell[2]:
                split = 1 # bca* (acb)
            else:
                split = 2 # bac* (cab) # This should not occur
        else:
            # a* < b*
            if reciprocal_reindexed_unit_cell[1] <= reciprocal_reindexed_unit_cell[2]:
                split = 3 # abc* (cba) # This should not occur
            elif reciprocal_reindexed_unit_cell[0] >= reciprocal_reindexed_unit_cell[2]:
                split = 4 # cab* (bac)
            else:
                split = 5 # acb* (bca) # This should not occur
    elif lattice_system == 'orthorhombic':
        if reindexed_spacegroup_symbol_hm[0] in ['P', 'I', 'F']:
            split = 0
        else:
            if reciprocal_reindexed_unit_cell[0] <= reciprocal_reindexed_unit_cell[2]:
                if reciprocal_reindexed_unit_cell[1] <= reciprocal_reindexed_unit_cell[2]:
                    split = 0 # abc
                else:
                    split = 1 # acb
            else:
                split = 2 # cab
    return split


def reindex_entry_monoclinic(unit_cell, spacegroup_symbol, space='direct'):
    # This version reindexes the monoclinic entries to nonconventional settings
    #   SG # | Setting
    #      3 | P 1 2 1
    #      4 | P 1 21 1
    #      5 | I 1 2 1
    #      6 | P 1 m 1
    #      7 | P 1 n 1
    #      8 | I 1 m 1
    #      9 | I 1 a 1
    #     10 | P 1 2/m 1
    #     11 | P 1 21/m 1
    #     12 | I 1 2/m 1
    #     13 | P 1 2/n 1
    #     14 | P 1 21/n 1
    #     15 | I 1 2/a 1
    # Useful resources:
    #    http://pd.chem.ucl.ac.uk/pdnn/symm4/practice.htm
    #    https://onlinelibrary.wiley.com/iucr/itc/Ab/ch5o1v0001/sec5o1o3.pdf
    def reindex_unit_cell(unit_cell, operator):
        a = unit_cell[0]
        b = unit_cell[1]
        c = unit_cell[2]
        beta = unit_cell[4]
        ucm = np.array([
            [a, 0, c*np.cos(beta)],
            [0, b, 0],
            [0, 0, c*np.sin(beta)],
            ])
        rucm = ucm @ operator
        reindexed_unit_cell = np.zeros(6)
        reindexed_unit_cell[0] = np.linalg.norm(rucm[:, 0])
        reindexed_unit_cell[1] = np.linalg.norm(rucm[:, 1])
        reindexed_unit_cell[2] = np.linalg.norm(rucm[:, 2])
        dot_product = np.dot(rucm[:, 0], rucm[:, 2])
        mag = reindexed_unit_cell[0] * reindexed_unit_cell[2]
        reindexed_unit_cell[4] = np.arccos(dot_product / mag)
        reindexed_unit_cell[3] = np.pi/2
        reindexed_unit_cell[5] = np.pi/2
        return reindexed_unit_cell

    A_centered = ['A 1 2 1', 'A 1 m 1', 'A 1 a 1', 'A 1 2/m 1', 'A 1 2/a 1']
    C_centered = ['C 1 2 1', 'C 1 m 1', 'C 1 c 1', 'C 1 2/m 1', 'C 1 2/c 1']
    I_centered = ['I 1 2 1', 'I 1 m 1', 'I 1 a 1', 'I 1 2/m 1', 'I 1 2/a 1']
    P_symbols = ['P 1 2 1', 'P 1 m 1', 'P 1 2/m 1', 'P 1 21 1', 'P 1 21/m 1']
    Pa_symbols = ['P 1 a 1', 'P 1 2/a 1', 'P 1 21/a 1']
    Pc_symbols = ['P 1 c 1', 'P 1 2/c 1', 'P 1 21/c 1']
    Pn_symbols = ['P 1 n 1', 'P 1 2/n 1', 'P 1 21/n 1']
    standard_settings = P_symbols + Pn_symbols + I_centered
    
    reindexed_spacegroup_symbol = copy.copy(spacegroup_symbol)
    if spacegroup_symbol in standard_settings:
        centered_reindexer = np.eye(3)
    elif spacegroup_symbol in A_centered:
        centered_reindexer = np.array([
            [-1, 0, 1],
            [0, 1, 0],
            [-1, 0, 0],
            ])
        reindexed_spacegroup_symbol = reindexed_spacegroup_symbol.replace('A', 'I')
    elif spacegroup_symbol in C_centered:
        centered_reindexer = np.array([
            [0, 0, -1],
            [0, 1, 0],
            [1, 0, -1],
            ])
        reindexed_spacegroup_symbol = reindexed_spacegroup_symbol.replace('C', 'I')
        reindexed_spacegroup_symbol = reindexed_spacegroup_symbol.replace('c', 'a')
    elif spacegroup_symbol in ['I 1 2/c 1', 'I 1 c 1']:
        centered_reindexer = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0],
            ])
        reindexed_spacegroup_symbol = reindexed_spacegroup_symbol.replace('c', 'a')
    elif spacegroup_symbol in Pa_symbols:
        centered_reindexer = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            ])
        reindexed_spacegroup_symbol = reindexed_spacegroup_symbol.replace('a', 'n')
    elif spacegroup_symbol in Pc_symbols:
        centered_reindexer = np.array([
            [1, 0, -1],
            [0, 1, 0],
            [1, 0, 0],
            ])
        reindexed_spacegroup_symbol = reindexed_spacegroup_symbol.replace('c', 'n')
    else:
        # These cases include entries that are reported in settings:
        # 'A 1 n 1', 'C 1 n 1', 'C 1 2/n 1', 'A 1 2/n 1',
        # 'P 1 1 21/n', 'P 1 1 21' 'I 2/b 1 1', 'P 21/c 1 1', 'C 2/m 1 1', 'P 21/m 1 1'
        # None of these would reindex to settings with limited entries.
        return None, None, np.eye(3)

    reindexed_unit_cell = reindex_unit_cell(unit_cell, centered_reindexer)
    swap_ac = False
    if space == 'direct' and reindexed_unit_cell[0] > reindexed_unit_cell[2]:
        swap_ac = True
    if space == 'reciprocal' and reindexed_unit_cell[0] < reindexed_unit_cell[2]:
        swap_ac = True
    if swap_ac:
        ac_reindexer = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0],
            ])
        centered_reindexer = centered_reindexer @ ac_reindexer
        reindexed_unit_cell = reindex_unit_cell(reindexed_unit_cell, ac_reindexer)

    reindex_angle = False
    if space == 'direct' and reindexed_unit_cell[4] < np.pi/2:
        reindex_angle = True
    if space == 'reciprocal' and reindexed_unit_cell[4] > np.pi/2:
        reindex_angle = True

    if reindex_angle:
        obtuse_reindexer = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
            ])
        reindexed_unit_cell[4] = np.pi - reindexed_unit_cell[4]
        hkl_reindexer = centered_reindexer @ obtuse_reindexer
    else:
        hkl_reindexer = centered_reindexer
    return reindexed_unit_cell, reindexed_spacegroup_symbol, hkl_reindexer


def get_different_monoclinic_settings(unit_cell, partial_unit_cell=False):
    ac_reindexer = [
        np.eye(3),
        np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0],
            ])
        ]
    transformations = [
        np.eye(3),
        np.array([
            [-1, 0, 1],
            [0, 1, 0],
            [-1, 0, 0],
            ]),
        np.array([
            [0, 0, -1],
            [0, 1, 0],
            [1, 0, -1],
            ]),
        np.array([
            [1, 0, 0],
            [0, 1, 0],
            [-1, 0, 1],
            ]),
        np.array([
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 1],
            ]),
        ]
    if partial_unit_cell:
        ucm = np.array([
            [unit_cell[0], 0, unit_cell[2] * np.cos(unit_cell[3])],
            [0, unit_cell[1], 0],
            [0, 0, unit_cell[2] * np.sin(unit_cell[3])],
            ])
        reindexed_unit_cell = np.zeros((10, 4))
    else:
        ucm = np.array([
            [unit_cell[0], 0, unit_cell[2] * np.cos(unit_cell[4])],
            [0, unit_cell[1], 0],
            [0, 0, unit_cell[2] * np.sin(unit_cell[4])],
            ])
        reindexed_unit_cell = np.zeros((10, 6))
    i = 0
    for trans in transformations:
        for perm in ac_reindexer:
            rucm = ucm @ perm @ trans
            reindexed_unit_cell[i, 0] = np.linalg.norm(rucm[:, 0])
            reindexed_unit_cell[i, 1] = np.linalg.norm(rucm[:, 1])
            reindexed_unit_cell[i, 2] = np.linalg.norm(rucm[:, 2])
            dot_product = np.dot(rucm[:, 0], rucm[:, 2])
            mag = reindexed_unit_cell[i, 0] * reindexed_unit_cell[i, 2]
            beta = np.arccos(dot_product / mag)
            if partial_unit_cell:
                reindexed_unit_cell[i, 3] = beta
            else:
                reindexed_unit_cell[i, 3] = np.pi/2
                reindexed_unit_cell[i, 4] = beta
                reindexed_unit_cell[i, 5] = np.pi/2
            i += 1
    return reindexed_unit_cell


def monoclinic_standardization(unit_cell, partial_unit_cell=False):
    # This performs a very quick "standardization" of the monoclinic lattice.
    # It performs a Selling reduction and then places the unique axis at b
    if partial_unit_cell:
        unit_cell_full = np.zeros((unit_cell.shape[0], 6))
        unit_cell_full[:, :3] = unit_cell[:, :3]
        unit_cell_full[:, 4] = unit_cell[:, 3]
        unit_cell_full[:, 3] = np.pi/2
        unit_cell_full[:, 5] = np.pi/2
    else:
        unit_cell_full = unit_cell
    reduced_unit_cell_full, _, _ = selling_reduction(unit_cell_full)
    right_angle = np.isclose(reduced_unit_cell_full[:, 3:], np.pi/2)

    standardized_unit_cell_full = reduced_unit_cell_full.copy()
    for i in range(unit_cell.shape[0]):
        if right_angle[i, 0] and right_angle[i, 2]:
            # a -> a, b -> b, c -> c
            continue
        elif right_angle[i, 0] and right_angle[i, 1]:
            # a -> -a, b -> -c, c -> -b
            standardized_unit_cell_full[i, 1] = reduced_unit_cell_full[i, 2]
            standardized_unit_cell_full[i, 2] = reduced_unit_cell_full[i, 1]
            standardized_unit_cell_full[i, 4] = reduced_unit_cell_full[i, 5]
            standardized_unit_cell_full[i, 5] = np.pi/2
        elif right_angle[i, 1] and right_angle[i, 2]:
            # a -> -b, b -> -b, c -> -c
            standardized_unit_cell_full[i, 0] = reduced_unit_cell_full[i, 1]
            standardized_unit_cell_full[i, 1] = reduced_unit_cell_full[i, 0]
            standardized_unit_cell_full[i, 4] = reduced_unit_cell_full[i, 3]
            standardized_unit_cell_full[i, 3] = np.pi/2
        else:
            # This case should be fixed. 
            standardized_unit_cell_full[i] = unit_cell_full[i]
    if partial_unit_cell:
        return standardized_unit_cell_full[:, [0, 1, 2, 4]]
    else:
        return standardized_unit_cell_full


def get_s6_from_unit_cell(unit_cell):
    a = unit_cell[:, 0]
    b = unit_cell[:, 1]
    c = unit_cell[:, 2]
    alpha = unit_cell[:, 3]
    beta = unit_cell[:, 4]
    gamma = unit_cell[:, 5]

    ax = a
    bx = b*np.cos(gamma)
    by = b*np.sin(gamma)
    cx = c*np.cos(beta)
    arg = (np.cos(alpha) - np.cos(beta)*np.cos(gamma)) / np.sin(gamma)
    cy = c * arg
    cz = c * np.sqrt(np.sin(beta)**2 - arg**2)
    z = np.zeros(unit_cell.shape[0])
    om = np.array([
        [ax, bx, cx],
        [z,  by, cy],
        [z,  z,  cz]
        ])
    om = np.moveaxis(om, [0, 1, 2], [1, 2, 0])
    d = -np.sum(om, axis=2)

    s6 = np.column_stack((
        np.sum(om[:, :, 1] * om[:, :, 2], axis=1),
        np.sum(om[:, :, 0] * om[:, :, 2], axis=1),
        np.sum(om[:, :, 0] * om[:, :, 1], axis=1),
        np.sum(om[:, :, 0] * d, axis=1),
        np.sum(om[:, :, 1] * d, axis=1),
        np.sum(om[:, :, 2] * d, axis=1),
        ))
    return s6


def get_unit_cell_from_s6_with_warnings(s6):
    a = np.sqrt(-(s6[:, 3] + s6[:, 1] + s6[:, 2]))
    b = np.sqrt(-(s6[:, 4] + s6[:, 0] + s6[:, 2]))
    c = np.sqrt(-(s6[:, 5] + s6[:, 0] + s6[:, 1]))
    alpha = np.arccos(s6[:, 0] / (b*c))
    beta = np.arccos(s6[:, 1] / (a*c))
    gamma = np.arccos(s6[:, 2] / (a*b))

    unit_cell = np.column_stack((a, b, c, alpha, beta, gamma))
    return unit_cell


def get_unit_cell_from_s6(s6):
    a2 = -(s6[:, 3] + s6[:, 1] + s6[:, 2])
    b2 = -(s6[:, 4] + s6[:, 0] + s6[:, 2])
    c2 = -(s6[:, 5] + s6[:, 0] + s6[:, 1])
    nonphysical_lengths = np.any(np.stack((a2, b2, c2), axis=1) <= 0, axis=1)
    if np.count_nonzero(nonphysical_lengths) == 0:
        a = np.sqrt(a2)
        b = np.sqrt(b2)
        c = np.sqrt(c2)
        alpha_arg = s6[:, 0] / (b*c)
        beta_arg = s6[:, 1] / (a*c)
        gamma_arg = s6[:, 2] / (a*b)
        nonphysical_angles = np.any(
            np.abs(np.stack((alpha_arg, beta_arg, gamma_arg), axis=1)) > 1,
            axis=1
        )
    else:
        physical_lengths = np.invert(nonphysical_lengths)
        a = np.zeros(s6.shape[0])
        b = np.zeros(s6.shape[0])
        c = np.zeros(s6.shape[0])

        a[nonphysical_lengths] = np.nan
        b[nonphysical_lengths] = np.nan
        c[nonphysical_lengths] = np.nan

        a[physical_lengths] = np.sqrt(a2[physical_lengths])
        b[physical_lengths] = np.sqrt(b2[physical_lengths])
        c[physical_lengths] = np.sqrt(c2[physical_lengths])

        alpha_arg = np.zeros(s6.shape[0])
        beta_arg = np.zeros(s6.shape[0])
        gamma_arg = np.zeros(s6.shape[0])

        alpha_arg[nonphysical_lengths] = np.nan
        beta_arg[nonphysical_lengths] = np.nan
        gamma_arg[nonphysical_lengths] = np.nan

        alpha_arg[physical_lengths] = s6[physical_lengths, 0] / (b*c)[physical_lengths]
        beta_arg[physical_lengths] = s6[physical_lengths, 1] / (a*c)[physical_lengths]
        gamma_arg[physical_lengths] = s6[physical_lengths, 2] / (a*b)[physical_lengths]

        nonphysical_angles = nonphysical_lengths

        nonphysical_angles[physical_lengths] = np.any(np.abs(np.stack((
            alpha_arg[physical_lengths], 
            beta_arg[physical_lengths], 
            gamma_arg[physical_lengths]
            ), axis=1)) > 1, axis=1)

    if np.count_nonzero(nonphysical_angles) == 0:
        alpha = np.arccos(alpha_arg)
        beta = np.arccos(beta_arg)
        gamma = np.arccos(gamma_arg)
    else:
        physical_angles = np.invert(nonphysical_angles)
        alpha = np.zeros(s6.shape[0])
        beta = np.zeros(s6.shape[0])
        gamma = np.zeros(s6.shape[0])

        alpha[nonphysical_angles] = np.nan
        beta[nonphysical_angles] = np.nan
        gamma[nonphysical_angles] = np.nan

        alpha[physical_angles] = np.arccos(alpha_arg[physical_angles])
        beta[physical_angles] = np.arccos(beta_arg[physical_angles])
        gamma[physical_angles] = np.arccos(gamma_arg[physical_angles])

    unit_cell = np.column_stack((a, b, c, alpha, beta, gamma))
    return unit_cell


def selling_reduction(unit_cell, space='direct'):
    assert space == 'direct'

    reduction_op_bc = np.array([
        [-1, 0, 0, 0, 0, 0], 
        [1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [-1, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1],
        ])
    reduction_op_bc_hkl = np.array([
        [1, 0, 0],
        [1, -1, 0],
        [0, 0, 1],
        ])

    reduction_op_ac = np.array([
        [1, 1, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, -1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1],
        ])
    reduction_op_ac_hkl = np.array([
        [1, -1, 0],
        [0, -1, 0],
        [0, 0, -1],
        ])

    reduction_op_ab = np.array([
        [1, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, -1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, -1, 0, 0, 1],
        ])
    reduction_op_ab_hkl = np.array([
        [1, 0, -1],
        [0, -1, 0],
        [0, 0, -1],
        ])

    reduction_op_ad = np.array([
        [1, 0, 0, -1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 0, -1, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 1],
        ])
    reduction_op_ad_hkl = np.array([
        [1, -1, -1],
        [0, -1, 0],
        [0, 0, -1],
        ])

    reduction_op_bd = np.array([
        [0, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, -1, 0],
        [1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, -1, 0],
        [0, 0, 0, 0, 1, 1],
        ])
    reduction_op_bd_hkl = np.array([
        [-1, 0, 0],
        [-1, 1, -1],
        [0, 0, -1],
        ])

    reduction_op_cd = np.array([
        [0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, -1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, -1],
        ])
    reduction_op_cd_hkl = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [-1, -1, 1],
        ])

    reduction_ops = [
        reduction_op_bc,
        reduction_op_ac,
        reduction_op_ab,
        reduction_op_ad,
        reduction_op_bd,
        reduction_op_cd
        ]
    reduction_ops_hkl = [
        reduction_op_bc_hkl,
        reduction_op_ac_hkl,
        reduction_op_ab_hkl,
        reduction_op_ad_hkl,
        reduction_op_bd_hkl,
        reduction_op_cd_hkl
        ]

    reflections = np.stack([
        np.eye(6),
        np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            ]),
        np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            ]),
        np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            ]),
        np.array([
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1],
            ]),
        np.array([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            ]),
        np.array([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            ]),
        np.array([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            ]),
        np.array([
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            ]),
        np.array([
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            ]),
        np.array([
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            ]),
        np.array([
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            ]),
        np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
            ]),
        np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0],
            ]),
        np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            ]),
        np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            ]),
        np.array([
            [0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            ]),
        np.array([
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0],
            ]),
        np.array([
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            ]),
        np.array([
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            ]),
        np.array([
            [0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            ]),
        np.array([
            [0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            ]),
        np.array([
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            ]),
        np.array([
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            ]),
        ],
        axis=0
        )

    reflection_op_abc = reflections[0]
    reflection_op_acb = reflections[1]
    reflection_op_bac = reflections[4]
    reflection_op_bca = reflections[5]
    reflection_op_cab = reflections[8]
    reflection_op_cba = reflections[9]

    reflection_op_abc_hkl = np.eye(3) #012
    reflection_op_acb_hkl = np.array([ #021
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        ])
    reflection_op_bac_hkl = np.array([ #102
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        ])
    reflection_op_bca_hkl = np.array([ #120
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        ])
    reflection_op_cab_hkl = np.array([ #201
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        ])
    reflection_op_cba_hkl = np.array([ #210
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        ])

    s6 = get_s6_from_unit_cell(unit_cell)
    s6_reduced = s6.copy()
    hkl_transformation = np.repeat(np.eye(3)[np.newaxis], unit_cell.shape[0], axis=0)
    for iteration in range(20):
        s6_reduced_next = s6_reduced.copy()
        hkl_transformation_next = hkl_transformation.copy()
        s6_max_index = np.argmax(s6_reduced, axis=1)
        s6_max = np.take_along_axis(s6_reduced, s6_max_index[:, np.newaxis], axis=1)[:, 0]
        for axis_index in range(6):
            indices = np.logical_and(
                s6_max_index == axis_index,
                s6_max > 0
                )
            if indices.sum() > 0:
                s6_reduced_next[indices] = np.matmul(
                    reduction_ops[axis_index],
                    s6_reduced[indices][:, :, np.newaxis]
                    )[:, :, 0]
                hkl_transformation_next[indices] = np.matmul(
                    hkl_transformation[indices],
                    reduction_ops_hkl[axis_index][np.newaxis]
                    )
        indices = s6_reduced_next.sum(axis=1) >= s6_reduced.sum(axis=1)
        s6_reduced[indices] = s6_reduced_next[indices]
        hkl_transformation[indices] =  hkl_transformation_next[indices]

    # There is a numerical warning for invalid value encountered in sqrt.
    # Removing the square root should work and not produce the warning.
    order = np.argsort(np.column_stack((
        np.sqrt(-(s6_reduced[:, 3] + s6_reduced[:, 1] + s6_reduced[:, 2])),
        np.sqrt(-(s6_reduced[:, 4] + s6_reduced[:, 0] + s6_reduced[:, 2])),
        np.sqrt(-(s6_reduced[:, 5] + s6_reduced[:, 0] + s6_reduced[:, 1]))
        )), axis=1)
    #order = np.argsort(np.column_stack((
    #    -(s6_reduced[:, 3] + s6_reduced[:, 1] + s6_reduced[:, 2]),
    #    -(s6_reduced[:, 4] + s6_reduced[:, 0] + s6_reduced[:, 2]),
    #    -(s6_reduced[:, 5] + s6_reduced[:, 0] + s6_reduced[:, 1])
    #    )), axis=1)

    abc = np.all(order == np.array([0, 1, 2]), axis=1)
    acb = np.all(order == np.array([0, 2, 1]), axis=1)
    bac = np.all(order == np.array([1, 0, 2]), axis=1)
    bca = np.all(order == np.array([1, 2, 0]), axis=1)
    cab = np.all(order == np.array([2, 0, 1]), axis=1)
    cba = np.all(order == np.array([2, 1, 0]), axis=1)

    s6_reduced[abc] = (reflection_op_abc @ s6_reduced[abc][:, :, np.newaxis])[:, :, 0]
    s6_reduced[acb] = (reflection_op_acb @ s6_reduced[acb][:, :, np.newaxis])[:, :, 0]
    s6_reduced[bac] = (reflection_op_bac @ s6_reduced[bac][:, :, np.newaxis])[:, :, 0]
    s6_reduced[bca] = (reflection_op_bca @ s6_reduced[bca][:, :, np.newaxis])[:, :, 0]
    s6_reduced[cab] = (reflection_op_cab @ s6_reduced[cab][:, :, np.newaxis])[:, :, 0]
    s6_reduced[cba] = (reflection_op_cba @ s6_reduced[cba][:, :, np.newaxis])[:, :, 0]

    hkl_transformation[abc] = hkl_transformation[abc] @ reflection_op_abc_hkl[np.newaxis]
    hkl_transformation[acb] = hkl_transformation[acb] @ reflection_op_acb_hkl[np.newaxis]
    hkl_transformation[bac] = hkl_transformation[bac] @ reflection_op_bac_hkl[np.newaxis]
    hkl_transformation[bca] = hkl_transformation[bca] @ reflection_op_bca_hkl[np.newaxis]
    hkl_transformation[cab] = hkl_transformation[cab] @ reflection_op_cab_hkl[np.newaxis]
    hkl_transformation[cba] = hkl_transformation[cba] @ reflection_op_cba_hkl[np.newaxis]

    unit_cell_reduced = get_unit_cell_from_s6(s6_reduced)
    return unit_cell_reduced, hkl_transformation, s6_reduced


def reindex_entry_triclinic(unit_cell, space='direct'):
    if unit_cell.shape == (6,):
        unit_cell_reduced, hkl_transformation, s6_reduced = selling_reduction(unit_cell[np.newaxis], space)
        unit_cell_reduced = unit_cell_reduced[0]
        hkl_transformation = hkl_transformation[0]
    else:
        unit_cell_reduced, hkl_transformation, s6_reduced = selling_reduction(unit_cell, space)
    return unit_cell_reduced, hkl_transformation


def reindex_entry_basic(unit_cell, lattice_system, bravais_lattice, space='direct'):
    """
    This function is meant to be called during optimization for a quick reindexing. There is an
    assumption that the unit cell has been placed in the correct setting. For example, all the
    centered monoclinic entries are initially placed in a body centered setting by 
    reindex_entry_monoclinic. reindex_entry_monoclinic_basic assumes this has already been performed.

    These also have been set so they operate in a vectorized manner on partial unit cells
    """
    if lattice_system == 'orthorhombic':
        if bravais_lattice == 'oC':
            order = np.argsort(unit_cell[:, :2], axis=1)
            if space == 'reciprocal':
                order = order[:, ::-1]
            unit_cell[:, :2] = np.take_along_axis(unit_cell[:, :2], order, axis=1)
        else:
            order = np.argsort(unit_cell, axis=1)
            if space == 'reciprocal':
                order = order[:, ::-1]
            unit_cell = np.take_along_axis(unit_cell, order, axis=1)
    elif lattice_system == 'monoclinic':
        if space == 'direct':
            swap_ac = unit_cell[:, 0] > unit_cell[:, 2]
            mirror_angle = unit_cell[:, 3] < np.pi/2
        elif space == 'reciprocal':
            swap_ac = unit_cell[:, 0] < unit_cell[:, 2]
            mirror_angle = unit_cell[:, 3] > np.pi/2
        unit_cell[swap_ac] = np.take(unit_cell[swap_ac], [2, 1, 0, 3], axis=1)
        unit_cell[mirror_angle, 3] = np.pi - unit_cell[mirror_angle, 3]
    elif lattice_system == 'triclinic':
        if space == 'reciprocal':
            # The reciprocal space reindexing algorithm for triclinic is broken...
            # This is here until it gets fixed.
            from Utilities import reciprocal_uc_conversion
            direct_unit_cell = reciprocal_uc_conversion(
                unit_cell, partial_unit_cell=True, lattice_system='triclinic'
                )
            direct_unit_cell, _ = reindex_entry_triclinic(direct_unit_cell, 'direct')
            unit_cell = reciprocal_uc_conversion(
                direct_unit_cell, partial_unit_cell=True, lattice_system='triclinic'
                )
        else:
            unit_cell, _ = reindex_entry_triclinic(unit_cell, space)
    return unit_cell
