import numpy as np


def reindex_entry_orthorhombic(unit_cell, spacegroup_symbol, spacegroup_number):
    permutation, permuter = get_permutation(unit_cell)
    permuted_unit_cell = np.concatenate((np.matmul(np.abs(permuter), unit_cell[:3]), unit_cell[3:]))
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
    if isinstance(spacegroup_map_table[spacegroup_number], str):
        permuted_spacegroup_symbol = spacegroup_symbol
    elif isinstance(spacegroup_map_table[spacegroup_number], list):
        current_index = spacegroup_map_table[spacegroup_number].index(spacegroup_symbol)
        if current_index == 0:
            # abc
            if permutation == 'abc':
                new_index = 0
            elif permutation == 'acb':
                new_index = 1
            elif permutation == 'bac':
                new_index = 2
            elif permutation == 'bca':
                new_index = 3
            elif permutation == 'cab':
                new_index = 4
            elif permutation == 'cba':
                new_index = 5
        elif current_index == 1:
            # acb
            if permutation == 'abc':
                new_index = 1
            elif permutation == 'acb':
                new_index = 0
            elif permutation == 'bac':
                new_index = 4
            elif permutation == 'bca':
                new_index = 5
            elif permutation == 'cab':
                new_index = 2
            elif permutation == 'cba':
                new_index = 3
        elif current_index == 2:
            # bac
            if permutation == 'abc':
                new_index = 2
            elif permutation == 'acb':
                new_index = 3
            elif permutation == 'bac':
                new_index = 0
            elif permutation == 'bca':
                new_index = 1
            elif permutation == 'cab':
                new_index = 5
            elif permutation == 'cba':
                new_index = 4
        elif current_index == 3:
            # bca
            if permutation == 'abc':
                new_index = 3
            elif permutation == 'acb':
                new_index = 2
            elif permutation == 'bac':
                new_index = 5
            elif permutation == 'bca':
                new_index = 4
            elif permutation == 'cab':
                new_index = 0
            elif permutation == 'cba':
                new_index = 1
        elif current_index == 4:
            # cab
            if permutation == 'abc':
                new_index = 4
            elif permutation == 'acb':
                new_index = 5
            elif permutation == 'bac':
                new_index = 1
            elif permutation == 'bca':
                new_index = 0
            elif permutation == 'cab':
                new_index = 3
            elif permutation == 'cba':
                new_index = 2
        elif current_index == 5:
            # cba
            if permutation == 'abc':
                new_index = 5
            elif permutation == 'acb':
                new_index = 4
            elif permutation == 'bac':
                new_index = 3
            elif permutation == 'bca':
                new_index = 2
            elif permutation == 'cab':
                new_index = 1
            elif permutation == 'cba':
                new_index = 0
        permuted_spacegroup_symbol = spacegroup_map_table[spacegroup_number][new_index]
    return permuted_spacegroup_symbol, permutation, permuted_unit_cell

def reindex_entry_orthorhombic_old(unit_cell, spacegroup_symbol):
    """
    This is really hacked in just to see if it helps in the regression task. Lots of guesses were made
    Should be generalized to other lattice systems.
    """
    permutation, permuter = get_permutation(unit_cell)
    permuted_unit_cell = np.concatenate((np.matmul(permuter, unit_cell[:3]), unit_cell[3:]))
    permuted_spacegroup_symbol = None
    # Cases where swapping the axes does not affect the systematic absences
    symmetric_list = [
        'P222', 'Pmm2', 'Pm2m', 'P2mm', 'Pmmm', 'P212121', 'Pnnn', 'Pbca', 'Pcab',
        'I222', 'I2mm', 'Im2m', 'Imm2', 'Immm', 'I212121', 'Ibca',  'Icab',
        'F222', 'Fmm2', 'Fmmm', 'Fddd', 'Fm2m',
        ]
    if spacegroup_symbol in symmetric_list:
        permuted_spacegroup_symbol = spacegroup_symbol

    # Cases where only the movement of one axes matters.
    # hOO_condition: if spacegroup_symbol is in this list, what matters is how the a axis transforms
    #   if in hOO_condition
    #       if a remains the same, spacegroup_symbol remains the same
    #       if a moves to b, pick spacegroup_symbol at same index in OkO_condition
    #       if a moves to c, pick spacegroup_symbol at same index in OOl_condition
    # spacegroup                                                                                     18       29       57      32      55
    hOO_condition_P0 = ['P2122', 'P21ma', 'Pm2a', 'Pmma', 'P21am', 'Pma2', 'Pmam', 'P2aa', 'Pmaa', 'P22121', 'Pbc21', 'Pbcm', 'P2cb', 'Pmcb']
    OkO_condition_P0 = ['P2212', 'Pm21b', 'P2mb', 'Pmmb', 'Pb21m', 'Pbm2', 'Pbmm', 'Pb2b', 'Pbmb', 'P21221', 'Pca21', 'Pcam', 'Pc2a', 'Pcma']
    OOl_condition_P0 = ['P2221', 'Pmc21', 'P2cm', 'Pmcm', 'Pcm21', 'Pc2m', 'Pcmm', 'Pcc2', 'Pccm', 'P21212', 'Pb21a', 'Pbma', 'Pba2', 'Pbam']
    # spacegroup          29       57      30      53      30      53       31      31       59      50      54      54      34
    hOO_condition_P1 = ['Pc21b', 'Pcmb', 'Pnc2', 'Pncm', 'Pn2b', 'Pnmb', 'Pnm21', 'Pn21m', 'Pnmm', 'Pncb', 'Pccb', 'Pbcb', 'P2nn']
    OkO_condition_P1 = ['P21ca', 'Pmca', 'Pcn2', 'Pcnm', 'P2na', 'Pmna', 'Pmn21', 'P21nm', 'Pmnm', 'Pcna', 'Pcca', 'Pcaa', 'Pn2n']
    OOl_condition_P1 = ['P21ab', 'Pmab', 'Pb2n', 'Pbmn', 'P2an', 'Pman', 'Pm21n', 'P21mn', 'Pmmn', 'Pban', 'Pbaa', 'Pbab', 'Pnn2']

    hOO_condition_F = ['F2dd']
    OkO_condition_F = ['Fd2d']
    OOl_condition_F = ['Fdd2']

    hOO_condition_I = ['I2aa', 'Imaa', 'Ibm2', 'Ib2m', 'Ibmm', 'I2mm', 'Icm2', 'I2cb', 'Iamm', 'Imca', 'Icmm', 'Imcb']
    OkO_condition_I = ['Ib2a', 'Ibma', 'I2am', 'Ima2', 'Imam', 'Im2m', 'I2cm', 'Ic2a', 'Imbm', 'Icma', 'Imcm', 'Icmb']
    OOl_condition_I = ['Iba2', 'Ibam', 'Im2a', 'I2ma', 'Imma', 'Imm2', 'Ic2m', 'Ica2', 'Immb', 'Icam', 'Immc', 'Icbm']

    hOO_condition = hOO_condition_P0 + hOO_condition_P1 + hOO_condition_F + hOO_condition_I
    OkO_condition = OkO_condition_P0 + OkO_condition_P1 + OkO_condition_F + OkO_condition_I
    OOl_condition = OOl_condition_P0 + OOl_condition_P1 + OOl_condition_F + OOl_condition_I
    if spacegroup_symbol in hOO_condition:
        index = hOO_condition.index(spacegroup_symbol)
        if permutation[0] == 'a':
            permuted_spacegroup_symbol = spacegroup_symbol
        elif permutation[1] == 'a':
            permuted_spacegroup_symbol = OkO_condition[index]
        elif permutation[2] == 'a':
            permuted_spacegroup_symbol = OOl_condition[index]
    elif spacegroup_symbol in OkO_condition:
        index = OkO_condition.index(spacegroup_symbol)
        if permutation[0] == 'b':
            permuted_spacegroup_symbol = hOO_condition[index]
        elif permutation[1] == 'b':
            permuted_spacegroup_symbol = spacegroup_symbol
        elif permutation[2] == 'b':
            permuted_spacegroup_symbol = OOl_condition[index]
    elif spacegroup_symbol in OOl_condition:
        index = OOl_condition.index(spacegroup_symbol)
        if permutation[0] == 'c':
            permuted_spacegroup_symbol = hOO_condition[index]
        elif permutation[1] == 'c':
            permuted_spacegroup_symbol = OkO_condition[index]
        elif permutation[2] == 'c':
            permuted_spacegroup_symbol = spacegroup_symbol

    # cases where the movement of multiple axes matters.
    # These are the oC spacegroups.
    if permutation == 'abc':
        permuted_spacegroup_symbol = spacegroup_symbol
    #                                                                          -> Guessed from here
    # spacegroups 21, 35,     38,     38      39      39     67       20       37      66      40      36       63      40     63       40       41      64      41      64      68
    setA = ['A222', 'A2mm', 'Amm2', 'Am2m', 'Abm2', 'Ab2m', 'Abmm', 'A2122', 'A2aa', 'Amaa', 'Ama2', 'A21am', 'Amam', 'Ama2', 'Amma', 'A21ma', 'Aba2', 'Abam', 'Ab2a', 'Abma', 'Abaa', 'Ammm', 'Acam', 'Ac2a', 'Ab2m', 'Acaa', 'Aabm', 'Aa2m']
    setB = ['B222', 'Bm2m', 'Bmm2', 'B2mm', 'Bma2', 'B2am', 'Bmam', 'B2212', 'Bb2b', 'Bbmb', 'B2mb', 'Bm21b', 'Bmmb', 'Bbm2', 'Bbmm', 'Bb21m', 'Bba2', 'Bbam', 'B2ab', 'Bmab', 'Bbab', 'Bmmm', 'Bbcm', 'B2ba', 'B2bm', 'Bbcb', 'Bcbm', 'Bb2m']
    setC = ['C222', 'Cmm2', 'Cm2m', 'C2mm', 'Cm2a', 'C2ma', 'Cmma', 'C2221', 'Ccc2', 'Cccm', 'Ccc2', 'Ccm21', 'Ccmm', 'C2cm', 'Cmcm', 'Cmc21', 'Cc2a', 'Ccma', 'C2ca', 'Cmca', 'Ccca', 'Cmmm', 'Ccam', 'C2cb', 'C2mb', 'Cccb', 'Ccmb', 'Cc2m']
    if spacegroup_symbol in setA:
        index = setA.index(spacegroup_symbol)
        if permutation in ['abc', 'acb']:
            permuted_spacegroup_symbol = spacegroup_symbol
        elif permutation in ['bac', 'cab']:
            permuted_spacegroup_symbol = setB[index]
        elif permutation in ['cba', 'bca']:
            permuted_spacegroup_symbol = setC[index]
    elif spacegroup_symbol in setB:
        index = setB.index(spacegroup_symbol)
        if permutation in ['bac', 'bca']:
            permuted_spacegroup_symbol = setA[index]
        elif permutation in ['abc', 'cba']:
            permuted_spacegroup_symbol = spacegroup_symbol
        elif permutation in ['acb', 'cab']:
            permuted_spacegroup_symbol = setC[index]
    elif spacegroup_symbol in setC:
        index = setC.index(spacegroup_symbol)
        if permutation in ['cba', 'cab']:
            permuted_spacegroup_symbol = setA[index]
        elif permutation in ['acb', 'bca']:
            permuted_spacegroup_symbol = setB[index]
        elif permutation in ['abc', 'bac']:
            permuted_spacegroup_symbol = spacegroup_symbol

    # These are space group settings that permuting the axes will not change the group for regression
    # and I did not feel like sorting out the new space group setting
    i_do_not_care_groups_0 = ['Pnma', 'Pna21', 'Pbna', 'Pcan', 'Pbcn', 'Pnab', 'Pnca', 'Pcnb', 'Pc21n', 'Pcmn', 'Pbn21']
    i_do_not_care_groups_1 = ['Pbnm', 'Pna21', 'Pnam', 'P21cn', 'Pmcn', 'P21nb', 'Pmnb', 'Pn21a', 'Pnma']
    i_do_not_care_groups_2 = ['Pnn21', 'Pnnm', 'Pn2n', 'Pnmn', 'P2nn', 'Pmnn', 'Pncn', 'Pnnb', 'Pnna', 'Pcnn', 'Pbnn', 'Pnan', 'Pnaa', 'Pbnb', 'Pccn']
    i_do_not_care_groups = i_do_not_care_groups_0 + i_do_not_care_groups_1 + i_do_not_care_groups_2
    if spacegroup_symbol in i_do_not_care_groups:
        permuted_spacegroup_symbol = spacegroup_symbol

    if permuted_spacegroup_symbol is None:
        permuted_spacegroup_symbol = spacegroup_symbol

    group = None
    if permuted_spacegroup_symbol in ['F222', 'Fmmm', 'Fmm2', 'Fm2m', 'F2mm']:
        group = 'oF_0'
    elif permuted_spacegroup_symbol in ['Fddd',	'Fdd2', 'Fd2d', 'F2dd']:
        group = 'oF_1'
    elif permuted_spacegroup_symbol in ['Immm', 'I222', 'I212121', 'Imm2', 'Im2m', 'I2mm']:
        group = 'oI_0'
    elif permuted_spacegroup_symbol in ['Ibca', 'Icab']:
        group = 'oI_1'
    elif permuted_spacegroup_symbol in ['I2aa', 'Imaa', 'Ib2a', 'Ibma', 'Iba2', 'Ibam', 'I2cb', 'Ic2a', 'Icma', 'Imcb', 'Icbm', 'Imca', 'Ica2']:
        group = 'oI_2'
    elif permuted_spacegroup_symbol in ['Ibm2', 'Ib2m', 'Ibmm', 'I2am', 'Ima2', 'Imam', 'Im2a', 'I2ma', 'Imma', 'I2cm', 'Ic2m', 'Immb', 'Imcm', 'Icm2', 'Icmm', 'Iamm', 'Imbm']:
        group = 'oI_3'
    elif permuted_spacegroup_symbol in ['C222', 'Cmm2', 'Cm2m', 'C2mm', 'Cmmm', 'Cm2a', 'C2ma', 'Cmma', 'C2mb']:
        group = 'oC_0'
    elif permuted_spacegroup_symbol in ['B222', 'Bm2m', 'Bmm2', 'B2mm', 'Bmmm', 'Bma2', 'B2am', 'Bmam', 'B2bm', 'Bb2m']:
        group = 'oC_1'
    elif permuted_spacegroup_symbol in ['A222', 'A2mm', 'Amm2', 'Am2m', 'Ammm', 'Abm2', 'Ab2m', 'Abmm']:
        group = 'oC_2'
    elif permuted_spacegroup_symbol in ['Cc2a', 'Ccma', 'C2ca', 'Cmca', 'Ccca', 'C2cb', 'Ccmb', 'Cc2b', 'Cccb', 'Ccam']:
        group = 'oC_3'
    elif permuted_spacegroup_symbol in ['Bba2', 'Bbam', 'B2ab', 'Bmab', 'Bbab', 'B2cb', 'B2ba', 'Bbcb', 'Bcbm']:
        group = 'oC_4'
    elif permuted_spacegroup_symbol in ['Aba2', 'Abam', 'Ab2a', 'Abma', 'Abaa', 'Acam', 'Ac2a', 'Acaa', 'Aabm']:
        group = 'oC_5'
    elif permuted_spacegroup_symbol in ['C2221', 'Ccc2', 'Cccm', 'Ccc2', 'Ccm21', 'Ccmm', 'C2cm', 'Cmcm', 'Cmc21', 'Cc2m']:
        group = 'oC_6'
    elif permuted_spacegroup_symbol in ['B2212', 'Bb2b', 'Bbmb', 'Bbm2', 'Bbmm', 'Bb21m', 'B2mb', 'Bm21b', 'Bmmb', 'Bbcm']:
        group = 'oC_7'
    elif permuted_spacegroup_symbol in ['A2122', 'A2aa', 'Amaa', 'Ama2', 'A21am', 'Amam', 'Ama2', 'Amma', 'A21ma']:
        group = 'oC_8'
    elif permuted_spacegroup_symbol in ['P222', 'Pmm2', 'Pm2m', 'P2mm', 'Pmmm']:
        group = 'oP_0'
    elif permuted_spacegroup_symbol in ['P2221', 'Pmc21', 'P2cm', 'Pmcm', 'Pcm21', 'Pc2m', 'Pcmm', 'Pcc2', 'Pccm']:
        group = 'oP_1'
    elif permuted_spacegroup_symbol in ['P2212', 'Pm21b', 'P2mb', 'Pmmb', 'Pb21m', 'Pbm2', 'Pbmm', 'Pb2b', 'Pbmb']:
        group = 'oP_2'
    elif permuted_spacegroup_symbol in ['P2122', 'P21ma', 'Pm2a', 'Pmma', 'P21am', 'Pma2', 'Pmam', 'P2aa', 'Pmaa']:
        group = 'oP_3'
    elif permuted_spacegroup_symbol in ['P21212', 'Pb21a', 'Pbma', 'Pba2', 'Pbam', 'P21ab', 'Pmab', 'Pb2n', 'Pbmn',	'P2an', 'Pman', 'Pm21n', 'P21mn', 'Pmmn', 'Pban', 'Pbaa', 'Pbab']:
        group = 'oP_4'
    elif permuted_spacegroup_symbol in ['P21221', 'Pca21', 'Pcam', 'Pc2a', 'Pcma', 'P21ca', 'Pmca', 'Pcn2', 'Pcnm', 'P2na', 'Pmna', 'Pmn21', 'P21nm', 'Pmnm', 'Pcna', 'Pcca', 'Pcaa']:
        group = 'oP_5'
    elif permuted_spacegroup_symbol in ['P22121', 'Pbc21', 'Pbcm', 'P2cb', 'Pmcb', 'Pc21b', 'Pcmb', 'Pnc2', 'Pncm', 'Pn2b', 'Pnmb', 'Pnm21', 'Pn21m', 'Pnmm', 'Pncb', 'Pccb', 'Pbcb']:
        group = 'oP_6'
    elif permuted_spacegroup_symbol in ['P212121', 'Pnnn', 'Pnn2', 'Pbca', 'Pcab', 'Pc21n', 'Pcmn', 'Pbn21', 'Pbnm', 'Pna21', 'Pnam', 'P21cn', 'Pmcn', 'P21nb', 'Pmnb', 'Pn21a', 'Pnma', 'Pnn21', 'Pnnm', 'Pn2n', 'Pnmn', 'P2nn', 'Pmnn', 'Pncn', 'Pnnb', 'Pnna', 'Pcnn', 'Pbnn', 'Pnan', 'Pnaa', 'Pbnb', 'Pccn', 'Pbna', 'Pcan', 'Pbcn', 'Pnab', 'Pnca', 'Pcnb']:
        group = 'oP_7'

    return permuted_unit_cell, permutation, permuted_spacegroup_symbol, group


def get_permutation(unit_cell):
    order = np.argsort(unit_cell[:3])
    if np.all(order == [0, 1, 2]):
        permutation = 'abc'
    elif np.all(order == [0, 2, 1]):
        permutation = 'acb'
    elif np.all(order == [1, 0, 2]):
        permutation = 'bac'
    elif np.all(order == [2, 0, 1]):
        permutation = 'bca'
    elif np.all(order == [1, 2, 0]):
        permutation = 'cab'
    elif np.all(order == [2, 1, 0]):
        permutation = 'cba'
    permuter = get_permuter(permutation)
    return permutation, permuter

def get_permuter(permutation):
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
    return permuter
