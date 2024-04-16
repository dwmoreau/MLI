import numpy as np


def reindex_entry(lattice_system, unit_cell, spacegroup_symbol, spacegroup_number):
    if lattice_system == 'orthorhombic':
        return reindex_entry_orthorhombic(unit_cell, spacegroup_symbol, spacegroup_number)
    elif lattice_system == 'monoclinic':
        return reindex_entry_monoclinic(unit_cell, spacegroup_symbol, spacegroup_number)


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
    return rhombohedral_unit_cell


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
    return permuted_spacegroup_symbol, permutation, permuted_unit_cell


def reindex_entry_monoclinic(unit_cell, spacegroup_symbol, spacegroup_number):
    permutation, permuter = get_permutation(unit_cell)
    permuted_unit_cell, permutation = permute_monoclinic(unit_cell, permutation, radians=False)
    #           'abc'       'acb',     'bac'       'bca'    'cab',       'cba'
    spacegroup_map_table = {
        '3i': ['P 1 2 1', 'P 1 1 2', 'P 2 1 1', 'P 2 1 1', 'P 1 1 2', 'P 1 2 1'],
        '3ii': ['B 1 2 1', 'C 1 1 2', 'A 2 1 1', 'A 2 1 1', 'C 1 1 2', 'B 1 2 1'],
        '4i': ['P 1 21 1', 'P 1 1 21', 'P 21 1 1', 'P 21 1 1', 'P 1 1 21', 'P 1 21 1'],
        '4ii': ['B 1 21 1', 'C 1 1 21', 'A 21 1 1', 'A 21 1 1', 'C 1 1 21', 'B 1 21 1'],
        '5i': ['C 1 2 1', 'A 1 1 2', 'B 2 1 1', 'B 2 1 1', 'A 1 1 2', 'C 1 2 1'],
        '5ii': ['A 1 2 1', 'B 1 1 2', 'C 2 1 1', 'C 2 1 1', 'B 1 1 2', 'A 1 2 1'],
        '5iii': ['I 1 2 1', 'I 1 1 2', 'I 2 1 1', 'I 2 1 1', 'I 1 1 2', 'I 1 2 1'],
        '5iv': ['F 1 2 1', 'F 1 1 2', 'F 2 1 1', 'F 2 1 1', 'F 1 1 2', 'F 1 2 1'],
        '6i': ['P 1 m 1', 'P 1 1 m', 'P m 1 1', 'P m 1 1', 'P 1 1 m', 'P 1 m 1'],
        '6ii': ['B 1 m 1', 'C 1 1 m', 'A m 1 1', 'A m 1 1', 'C 1 1 m', 'B 1 m 1'],
        '7i': ['P 1 c 1', 'P 1 1 a', 'P b 1 1', 'P b 1 1', 'P 1 1 a', 'P 1 c 1'],
        '7ii': ['P 1 a 1', 'P 1 1 b', 'P c 1 1', 'P c 1 1', 'P 1 1 b', 'P 1 a 1'],
        '7iii': ['P 1 n 1', 'P 1 1 n', 'P n 1 1', 'P n 1 1', 'P 1 1 n', 'P 1 n 1'],
        '7iv': ['B 1 a 1', 'C 1 1 a', 'A b 1 1', 'A b 1 1', 'C 1 1 a', 'B 1 a 1'],
        '7v': ['B 1 d 1', 'C 1 1 d', 'A d 1 1', 'A d 1 1', 'C 1 1 d', 'B 1 d 1'],
        '8i': ['C 1 m 1', 'A 1 1 m', 'B m 1 1', 'B m 1 1', 'A 1 1 m', 'C 1 m 1'],
        '8ii': ['A 1 m 1', 'B 1 1 m', 'C m 1 1', 'C m 1 1', 'B 1 1 m', 'A 1 m 1'],
        '8iii': ['I 1 m 1', 'I 1 1 m', 'I m 1 1', 'I m 1 1', 'I 1 1 m', 'I 1 m 1'],
        '8iv': ['F 1 m 1', 'F 1 1 m', 'F m 1 1', 'F m 1 1', 'F 1 1 m', 'F 1 m 1'],
        '9i': ['C 1 c 1', 'A 1 1 a', 'B b 1 1', 'B b 1 1', 'A 1 1 a', 'C 1 c 1'],
        '9ii': ['A 1 a 1', 'B 1 1 b', 'C c 1 1', 'C c 1 1', 'B 1 1 b', 'A 1 a 1'],
        '9iii': ['I 1 a 1', 'I 1 1 a', 'I b 1 1', 'I b 1 1', 'I 1 1 a', 'I 1 a 1'],
        '9iv': ['F 1 d 1', 'F 1 1 d', 'F d 1 1', 'F d 1 1', 'F 1 1 d', 'F 1 d 1'],
        '10i': ['P 1 2/m 1', 'P 1 1 2/m', 'P 2/m 1 1', 'P 2/m 1 1', 'P 1 1 2/m', 'P 1 2/m 1'],
        '10ii': ['B 1 2/m 1', 'C 1 1 2/m', 'A 2/m 1 1', 'A 2/m 1 1', 'C 1 1 2/m', 'B 1 2/m 1'],
        '11i': ['P 1 21/m 1', 'P 1 1 21/m', 'P 21/m 1 1', 'P 21/m 1 1', 'P 1 1 21/m', 'P 1 21/m 1'],
        '11ii': ['B 1 21/m 1', 'C 1 1 21/m', 'A 21/m 1 1', 'A 21/m 1 1', 'C 1 1 21/m', 'B 1 21/m 1'],
        '12i': ['C 1 2/m 1', 'A 1 1 2/m', 'B 2/m 1 1', 'B 2/m 1 1', 'A 1 1 2/m', 'C 1 2/m 1'],
        '12ii': ['A 1 2/m 1', 'B 1 1 2/m', 'C 2/m 1 1', 'C 2/m 1 1', 'B 1 1 2/m', 'A 1 2/m 1'],
        '12iii': ['I 1 2/m 1', 'I 1 1 2/m', 'I 2/m 1 1', 'I 2/m 1 1', 'I 1 1 2/m', 'I 1 2/m 1'],
        '12iv': ['F 1 2/m 1', 'F 1 1 2/m', 'F 2/m 1 1', 'F 2/m 1 1', 'F 1 1 2/m', 'F 1 2/m 1'],
        '13i': ['P 1 2/c 1', 'P 1 1 2/a', 'P 2/b 1 1', 'P 2/b 1 1', 'P 1 1 2/a', 'P 1 2/c 1'],
        '13ii': ['P 1 2/a 1', 'P 1 1 2/b', 'P 2/c 1 1', 'P 2/c 1 1', 'P 1 1 2/b', 'P 1 2/a 1'],
        '13iii': ['P 1 2/n 1', 'P 1 1 2/n', 'P 2/n 1 1', 'P 2/n 1 1', 'P 1 1 2/n', 'P 1 2/n 1'],
        '13iv': ['B 1 2/a 1', 'C 1 1 2/a', 'A 2/b 1 1', 'A 2/b 1 1', 'C 1 1 2/a', 'B 1 2/a 1'],
        '13v': ['B 1 2/d 1', 'C 1 1 2/d', 'A 2/d 1 1', 'A 2/d 1 1', 'C 1 1 2/d', 'B 1 2/d 1'],
        '14i': ['P 1 21/c 1', 'P 1 1 21/a', 'P 21/b 1 1', 'P 21/b 1 1', 'P 1 1 21/a', 'P 1 21/c 1'],
        '14ii': ['P 1 21/a 1', 'P 1 1 21/b', 'P 21/c 1 1', 'P 21/c 1 1', 'P 1 1 21/b', 'P 1 21/a 1'],
        '14iii': ['P 1 21/n 1', 'P 1 1 21/n', 'P 21/n 1 1', 'P 21/n 1 1', 'P 1 1 21/n', 'P 1 21/n 1'],
        '14iv': ['B 1 21/a 1', 'C 1 1 21/a', 'A 21/b 1 1', 'A 21/b 1 1', 'C 1 1 21/a', 'B 1 21/a 1'],
        '14v': ['B 1 21/d 1', 'C 1 1 21/d', 'A 21/d 1 1', 'A 21/d 1 1', 'C 1 1 21/d', 'B 1 21/d 1'],
        '15i': ['C 1 2/c 1', 'A 1 1 2/a', 'B 2/b 1 1', 'B 2/b 1 1', 'A 1 1 2/a', 'C 1 2/c 1'],
        '15ii': ['A 1 2/a 1', 'B 1 1 2/b', 'C 2/c 1 1', 'C 2/c 1 1', 'B 1 1 2/b', 'A 1 2/a 1'],
        '15iii': ['I 1 2/a 1', 'I 1 1 2/a', 'I 2/b 1 1', 'I 2/b 1 1', 'I 1 1 2/a', 'I 1 2/a 1'],
        '15iv': ['F 1 2/d 1', 'F 1 1 2/d', 'F 2/d 1 1', 'F 2/d 1 1', 'F 1 1 2/d', 'F 1 2/d 1'],
        }
    #           'abc'       'acb',     'bac'       'bca'    'cab',       'cba'
    for key in spacegroup_map_table.keys():
        if key.startswith(str(spacegroup_number)):
            if spacegroup_symbol in spacegroup_map_table[key]:
                map_table_key = key

    permuted_spacegroup_symbol = map_spacegroup_symbol(
        spacegroup_map_table, map_table_key, spacegroup_symbol, permutation
        )
    return permuted_spacegroup_symbol, permutation, permuted_unit_cell


def permute_monoclinic(unit_cell, permutation, radians):
    if radians:
        check = np.pi/2
    else:
        check = 90
    permuted_unit_cell = np.zeros(6)
    if permutation == 'abc':
        # without a copy here, if the angle is acute, permuted unit cell angle changes,
        # and so does unit cell angle
        permuted_unit_cell = unit_cell.copy()
        angle_index = 4
    elif permutation == 'acb':
        permuted_unit_cell = np.array([
            unit_cell[0],
            unit_cell[2],
            unit_cell[1],
            check,
            check,
            unit_cell[4],
            ])
        angle_index = 5
    elif permutation == 'bac':
        permuted_unit_cell = np.array([
            unit_cell[1],
            unit_cell[0],
            unit_cell[2],
            2*check - unit_cell[4],
            check,
            check,
            ])
        angle_index = 3
    elif permutation == 'bca':
        permuted_unit_cell = np.array([
            unit_cell[1],
            unit_cell[2],
            unit_cell[0],
            unit_cell[4],
            check,
            check,
            ])
        angle_index = 3
    elif permutation == 'cab':
        permuted_unit_cell = np.array([
            unit_cell[2],
            unit_cell[0],
            unit_cell[1],
            check,
            check,
            2*check - unit_cell[4],
            ])
        angle_index = 5
    elif permutation == 'cba':
        permuted_unit_cell = np.array([
            unit_cell[2],
            unit_cell[1],
            unit_cell[0],
            check,
            2*check - unit_cell[4],
            check,
            ])
        angle_index = 4
    # Any monoclinic unit cell can be represented with a obtuse or acute angle.
    # I am choosing to use the obtuse representation, this is consistent with most of the
    # entries in the ccdc.
    if permuted_unit_cell[angle_index] < check:
        # If acute, add a '<' (less than) sign to the permutation and make the angle obtuse.
        permutation += '<'
        permuted_unit_cell[angle_index] = 2*check - permuted_unit_cell[angle_index]
    else:
        permutation += '>'
    return permuted_unit_cell, permutation


def unpermute_monoclinic_full_unit_cell(permuted_unit_cell, permutation, radians):
    """
    the 'permutation' variable is the permutation that resulted in this unit cell
    The correct unpermurtation should give a unit cell angle of pi - angle in half of these
    cases. When the initial reindexing is performed, the angles are made obtuse and
    an additional operation is performed on the Miller indices.

    In otherwords, this should only be used to convert predicted unit cells in the reindexed
    representation to the original representation.
    """
    if radians:
        check = np.pi/2
    else:
        check = 90
    unit_cell = np.zeros(6)
    if permutation in ['abc<', 'abc>']:
        unit_cell = permuted_unit_cell.copy()
    elif permutation in ['acb<', 'acb>']:
        unit_cell = np.array([
            permuted_unit_cell[0],
            permuted_unit_cell[2],
            permuted_unit_cell[1],
            check,
            permuted_unit_cell[5],
            check,
            ])
    elif permutation in ['bac<', 'bac>']:
        unit_cell = np.array([
            permuted_unit_cell[1],
            permuted_unit_cell[0],
            permuted_unit_cell[2],
            check,
            permuted_unit_cell[3],
            check,
            ])
    elif permutation in ['bca<', 'bca>']:
        unit_cell = np.array([
            permuted_unit_cell[2],
            permuted_unit_cell[0],
            permuted_unit_cell[1],
            check,
            permuted_unit_cell[3],
            check,
            ])
    elif permutation in ['cab<', 'cab>']:
        unit_cell = np.array([
            permuted_unit_cell[1],
            permuted_unit_cell[2],
            permuted_unit_cell[0],
            check,
            permuted_unit_cell[5],
            check,
            ])
    elif permutation in ['cba<', 'cba>']:
        unit_cell = np.array([
            permuted_unit_cell[2],
            permuted_unit_cell[1],
            permuted_unit_cell[0],
            check,
            permuted_unit_cell[4],
            check,
            ])
    else:
        print('Failed unpermute')
        print(permutation)
        print(permuted_unit_cell)
        assert False
    return unit_cell


def unpermute_monoclinic_partial_unit_cell(permuted_unit_cell, permuted_unit_cell_cov, permutation, radians):
    """
    the 'permutation' variable is the permutation that resulted in this unit cell

    this should only be used to convert predicted unit cells in the reindexed
    representation to the original representation.
    """
    if radians:
        check = np.pi/2
    else:
        check = 90
    if not permuted_unit_cell is None:
        if permutation in ['abc<', 'abc>']:
            unit_cell = permuted_unit_cell.copy()
        elif permutation in ['acb<', 'acb>']:
            unit_cell = np.array([
                permuted_unit_cell[0],
                permuted_unit_cell[2],
                permuted_unit_cell[1],
                permuted_unit_cell[3],
                ])
        elif permutation in ['bac<', 'bac>']:
            unit_cell = np.array([
                permuted_unit_cell[1],
                permuted_unit_cell[0],
                permuted_unit_cell[2],
                permuted_unit_cell[3],
                ])
        elif permutation in ['bca<', 'bca>']:
            unit_cell = np.array([
                permuted_unit_cell[2],
                permuted_unit_cell[0],
                permuted_unit_cell[1],
                permuted_unit_cell[3],
                ])
        elif permutation in ['cab<', 'cab>']:
            unit_cell = np.array([
                permuted_unit_cell[1],
                permuted_unit_cell[2],
                permuted_unit_cell[0],
                permuted_unit_cell[3],
                ])
        elif permutation in ['cba<', 'cba>']:
            unit_cell = np.array([
                permuted_unit_cell[2],
                permuted_unit_cell[1],
                permuted_unit_cell[0],
                permuted_unit_cell[3],
                ])
        else:
            print('Failed unpermute')
            print(permutation)
            print(permuted_unit_cell)
            print()

    if not permuted_unit_cell_cov is None:
        if permutation in ['abc<', 'abc>']:
            unit_cell_cov = permuted_unit_cell_cov.copy()
        elif permutation in ['acb<', 'acb>']:
            unit_cell_cov = np.zeros((4, 4))
            unit_cell_cov[0, 0] = permuted_unit_cell_cov[0, 0]
            unit_cell_cov[1, 1] = permuted_unit_cell_cov[2, 2]
            unit_cell_cov[2, 2] = permuted_unit_cell_cov[1, 1]
            unit_cell_cov[3, 3] = permuted_unit_cell_cov[3, 3]
        elif permutation in ['bac<', 'bac>']:
            unit_cell_cov = np.zeros((4, 4))
            unit_cell_cov[0, 0] = permuted_unit_cell_cov[1, 1]
            unit_cell_cov[1, 1] = permuted_unit_cell_cov[0, 0]
            unit_cell_cov[2, 2] = permuted_unit_cell_cov[2, 2]
            unit_cell_cov[3, 3] = permuted_unit_cell_cov[3, 3]
        elif permutation in ['bca<', 'bca>']:
            unit_cell_cov = np.zeros((4, 4))
            unit_cell_cov[0, 0] = permuted_unit_cell_cov[2, 2]
            unit_cell_cov[1, 1] = permuted_unit_cell_cov[0, 0]
            unit_cell_cov[2, 2] = permuted_unit_cell_cov[1, 1]
            unit_cell_cov[3, 3] = permuted_unit_cell_cov[3, 3]
        elif permutation in ['cab<', 'cab>']:
            unit_cell_cov = np.zeros((4, 4))
            unit_cell_cov[0, 0] = permuted_unit_cell_cov[1, 1]
            unit_cell_cov[1, 1] = permuted_unit_cell_cov[2, 2]
            unit_cell_cov[2, 2] = permuted_unit_cell_cov[0, 0]
            unit_cell_cov[3, 3] = permuted_unit_cell_cov[3, 3]
        elif permutation in ['cba<', 'cba>']:
            unit_cell_cov = np.zeros((4, 4))
            unit_cell_cov[0, 0] = permuted_unit_cell_cov[2, 2]
            unit_cell_cov[1, 1] = permuted_unit_cell_cov[1, 1]
            unit_cell_cov[2, 2] = permuted_unit_cell_cov[0, 0]
            unit_cell_cov[3, 3] = permuted_unit_cell_cov[3, 3]
        else:
            print('Failed unpermute')
            print(permutation)
            print(permuted_unit_cell)
            print()

    if permuted_unit_cell is None and not permuted_unit_cell_cov is None:
        return unit_cell_cov
    elif not permuted_unit_cell is None and permuted_unit_cell_cov is None:
        return unit_cell
    elif not permuted_unit_cell is None and not permuted_unit_cell_cov is None:
        return unit_cell, unit_cell_cov


def unpermute_monoclinic_full_unit_cell_old(permuted_unit_cell, permutation, radians):
    """
    the 'permutation' variable is the permutation that resulted in this unit cell
    """
    if radians:
        check = np.pi/2
    else:
        check = 90
    unit_cell = np.zeros(6)
    if permutation == 'abc':
        unit_cell = permuted_unit_cell.copy()
    elif permutation == 'acb':
        unit_cell = np.array([
            permuted_unit_cell[0],
            permuted_unit_cell[2],
            permuted_unit_cell[1],
            check,
            permuted_unit_cell[5],
            check,
            ])
    elif permutation == 'bac':
        unit_cell = np.array([
            permuted_unit_cell[1],
            permuted_unit_cell[0],
            permuted_unit_cell[2],
            check,
            2*check - permuted_unit_cell[3],
            check,
            ])
    elif permutation == 'bca':
        unit_cell = np.array([
            permuted_unit_cell[2],
            permuted_unit_cell[0],
            permuted_unit_cell[1],
            check,
            permuted_unit_cell[3],
            check,
            ])
    elif permutation == 'cab':
        unit_cell = np.array([
            permuted_unit_cell[1],
            permuted_unit_cell[2],
            permuted_unit_cell[0],
            check,
            2*check - permuted_unit_cell[5],
            check,
            ])
    elif permutation == 'cba':
        unit_cell = np.array([
            permuted_unit_cell[2],
            permuted_unit_cell[1],
            permuted_unit_cell[0],
            check,
            2*check - permuted_unit_cell[4],
            check,
            ])
    return unit_cell


def unpermute_monoclinic_partial_unit_cell_old(permuted_unit_cell, permuted_unit_cell_cov, permutation, radians):
    """
    the 'permutation' variable is the permutation that resulted in this unit cell
    """
    if radians:
        check = np.pi/2
    else:
        check = 90
    if not permuted_unit_cell is None:
        if permutation == 'abc':
            unit_cell = permuted_unit_cell
        elif permutation == 'acb':
            unit_cell = np.array([
                permuted_unit_cell[0],
                permuted_unit_cell[2],
                permuted_unit_cell[1],
                permuted_unit_cell[3],
                ])
        elif permutation == 'bac':
            unit_cell = np.array([
                permuted_unit_cell[1],
                permuted_unit_cell[0],
                permuted_unit_cell[2],
                2 * check - permuted_unit_cell[3],
                ])
        elif permutation == 'bca':
            unit_cell = np.array([
                permuted_unit_cell[2],
                permuted_unit_cell[0],
                permuted_unit_cell[1],
                permuted_unit_cell[3],
                ])
        elif permutation == 'cab':
            unit_cell = np.array([
                permuted_unit_cell[1],
                permuted_unit_cell[2],
                permuted_unit_cell[0],
                2 * check - permuted_unit_cell[3],
                ])
        elif permutation == 'cba':
            unit_cell = np.array([
                permuted_unit_cell[2],
                permuted_unit_cell[1],
                permuted_unit_cell[0],
                2 * check - permuted_unit_cell[3],
                ])
    if not permuted_unit_cell_cov is None:
        if permutation == 'abc':
            unit_cell_cov = permuted_unit_cell_cov
        elif permutation == 'acb':
            unit_cell_cov = np.zeros((4, 4))
            unit_cell_cov[0, 0] = permuted_unit_cell_cov[0, 0]
            unit_cell_cov[1, 1] = permuted_unit_cell_cov[2, 2]
            unit_cell_cov[2, 2] = permuted_unit_cell_cov[1, 1]
            unit_cell_cov[3, 3] = permuted_unit_cell_cov[3, 3]
        elif permutation == 'bac':
            unit_cell_cov = np.zeros((4, 4))
            unit_cell_cov[0, 0] = permuted_unit_cell_cov[1, 1]
            unit_cell_cov[1, 1] = permuted_unit_cell_cov[0, 0]
            unit_cell_cov[2, 2] = permuted_unit_cell_cov[2, 2]
            unit_cell_cov[3, 3] = permuted_unit_cell_cov[3, 3]
        elif permutation == 'bca':
            unit_cell_cov = np.zeros((4, 4))
            unit_cell_cov[0, 0] = permuted_unit_cell_cov[2, 2]
            unit_cell_cov[1, 1] = permuted_unit_cell_cov[0, 0]
            unit_cell_cov[2, 2] = permuted_unit_cell_cov[1, 1]
            unit_cell_cov[3, 3] = permuted_unit_cell_cov[3, 3]
        elif permutation == 'cab':
            unit_cell_cov = np.zeros((4, 4))
            unit_cell_cov[0, 0] = permuted_unit_cell_cov[1, 1]
            unit_cell_cov[1, 1] = permuted_unit_cell_cov[2, 2]
            unit_cell_cov[2, 2] = permuted_unit_cell_cov[0, 0]
            unit_cell_cov[3, 3] = permuted_unit_cell_cov[3, 3]
        elif permutation == 'cba':
            unit_cell_cov = np.zeros((4, 4))
            unit_cell_cov[0, 0] = permuted_unit_cell_cov[2, 2]
            unit_cell_cov[1, 1] = permuted_unit_cell_cov[1, 1]
            unit_cell_cov[2, 2] = permuted_unit_cell_cov[0, 0]
            unit_cell_cov[3, 3] = permuted_unit_cell_cov[3, 3]

    if permuted_unit_cell is None and not permuted_unit_cell_cov is None:
        return unit_cell_cov
    elif not permuted_unit_cell is None and permuted_unit_cell_cov is None:
        return unit_cell
    elif not permuted_unit_cell is None and not permuted_unit_cell_cov is None:
        return unit_cell, unit_cell_cov


def make_monoclinic_obtuse(unit_cell, hkl=None, radians=True):
    if radians:
        check = np.pi/2
    else:
        check = 90
    reindexed_unit_cell = np.zeros(6)
    reindexed_unit_cell[:3] = unit_cell[:3]
    for index in range(3, 6):
        if unit_cell[index] != check:
            reindexed_unit_cell[index] = 2*check - unit_cell[index]
            if not hkl is None:
                if index in [3, 5]:
                    reindexed_hkl = hkl * np.array([-1, 1, -1])[np.newaxis]
                elif index == 4:
                    reindexed_hkl = hkl * np.array([-1, -1, 1])[np.newaxis]
        else:
            reindexed_unit_cell[index] = check
    if hkl is None:
        return reindexed_unit_cell
    else:
        return reindexed_unit_cell, reindexed_hkl
    

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
