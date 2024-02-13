import numpy as np


def calculate_s6(g6_params):
    """
    This was copied From Vanessa's code. I do not know how it works.
    This is detailed in the Andrews, Bernstein, & Sauter 2019 papers
    """
    #a = reduced_uc_params[0]
    #b = reduced_uc_params[1]
    #c = reduced_uc_params[2]
    #d = -1*(a+b+c)
    # [P, Q, R, S, T, U] = [s23, s13, s12, s14, s24, s34] = [bdotc, adotc, adotb, adotd, bdotd, cdotd]
    tol = 1e-10
    p = g6_params[3]/2.0
    q = g6_params[4]/2.0
    r = g6_params[5]/2.0
    s = (-2.0*g6_params[0] - g6_params[5] - g6_params[4]) / 2.0
    t = (-1.0*g6_params[5] - 2.0*g6_params[1] - g6_params[3]) / 2.0
    u = (-1.0*g6_params[4] - g6_params[3] - 2.0*g6_params[2]) / 2.0
    #tselling_vector = np.array([np.dot(b,c), np.dot(a,c), np.dot(a,b),
    #                           np.dot(a,d), np.dot(b,d), np.dot(c,d),
    #                          ]
    #                             )
    selling_vector = np.array([p, q, r, s, t, u])
    selling_vector = np.array([s if abs(s) > tol else 0 for s in selling_vector])
    #print(tselling_vector,selling_vector)
    reduction_matrices = np.array(
                                      [
                                       np.array(
                                                [
                                                 [-1, 0, 0, 0, 0, 0],
                                                 [ 1, 1, 0, 0, 0, 0],
                                                 [ 1, 0, 0, 0, 1, 0],
                                                 [-1, 0, 0, 1, 0, 0],
                                                 [ 1, 0, 1, 0, 0, 0],
                                                 [ 1, 0, 0, 0, 0, 1],
                                                ]
                                               ),
                                       np.array(
                                                [
                                                 [ 1, 1, 0, 0, 0, 0],
                                                 [ 0,-1, 0, 0, 0, 0],
                                                 [ 0, 1, 0, 1, 0, 0],
                                                 [ 0, 1, 1, 0, 0, 0],
                                                 [ 0,-1, 0, 0, 1, 0],
                                                 [ 0, 1, 0, 0, 0, 1],
                                                ]
                                               ),
                                       np.array(
                                                [
                                                 [ 1, 0, 1, 0, 0, 0],
                                                 [ 0, 0, 1, 1, 0, 0],
                                                 [ 0, 0,-1, 0, 0, 0],
                                                 [ 0, 1, 1, 0, 0, 0],
                                                 [ 0, 0, 1, 0, 1, 0],
                                                 [ 0, 0,-1, 0, 0, 1],
                                                ]
                                               ),
                                       np.array(
                                                [
                                                 [ 1, 0, 0,-1, 0, 0],
                                                 [ 0, 0, 1, 1, 0, 0],
                                                 [ 0, 1, 0, 1, 0, 0],
                                                 [ 0, 0, 0,-1, 0, 0],
                                                 [ 0, 0, 0, 1, 1, 0],
                                                 [ 0, 0, 0, 1, 0, 1],
                                                ]
                                               ),
                                     np.array(
                                              [
                                               [ 0, 0, 1, 0, 1, 0],
                                               [ 0, 1, 0, 0,-1, 0],
                                               [ 1, 0, 0, 0, 1, 0],
                                               [ 0, 0, 0, 1, 1, 0],
                                               [ 0, 0, 0, 0,-1, 0],
                                               [ 0, 0, 0, 0, 1, 1],
                                              ]
                                             ),
                                     np.array(
                                              [
                                               [ 0, 1, 0, 0, 0, 1],
                                               [ 1, 0, 0, 0, 0, 1],
                                               [ 0, 0, 1, 0, 0,-1],
                                               [ 0, 0, 0, 1, 0, 1],
                                               [ 0, 0, 0, 0, 1, 1],
                                               [ 0, 0, 0, 0, 0,-1],
                                              ]
                                             ),
                                  ]
                                 )

    while np.greater(np.max(selling_vector), 0):
        max_index = selling_vector.argmax()
        selling_vector = np.dot(reduction_matrices[max_index], selling_vector)
    #selling_vector = np.array([s if abs(s) > tol else 0 for s in selling_vector])

    if np.max(selling_vector) > 0:
        print("Selling vector reduction failed")    
        return None
    else:
        return selling_vector


def load_identifiers(file_name):
    identifiers = []
    with open(file_name, 'r') as identifiers_file:
        for row in identifiers_file:
            identifiers.append(row[:-1])
    return identifiers


def save_identifiers(file_name, identifiers):
    with open(file_name, 'w') as identifiers_file:
        for identifier in identifiers:
            identifiers_file.write(f'{identifier}\n')

def spacegroup_to_symmetry(space_group):
    # Values:
    #  0 - Bravais Lattice
    #  1 - Crystal family
    #  2 - Crystal system
    #  3 - Lattice system

    # I am using the wikipedia definitions
    #  https://en.wikipedia.org/wiki/Crystal_system
    #  Crystal Family:
    #    - triclinic
    #    - monoclinic
    #    - orthorhombic
    #    - tetragonal
    #    - hexagonal
    #    - cubic
    #
    # Crystal system    | Required symmetries
    #    - triclinic    | None
    #    - monoclinic   | 1 twofold axis of rotation 
    #                     or 1 mirror plane
    #    - orthorhombic | 3 twofold axes of rotation 
    #                     or 1 twofold axis of rotation and 2 mirror planes
    #    - tetragonal   | 1 fourfold axis of rotation
    #    - trigonal     | 1 threefold axis of rotation
    #    - hexagonal    | 1 sixfold axis of rotation
    #    - cubic        | 4 threefold axes of rotation
    #    hexagonal crystal family includes hexagonal and trigonal crystal systems 
    #    the rhombohedral lattice system is in the trigonal crystal system 
    #      and hexagonal crystal family
    # 
    # Bravais Lattices
    # P: primitive
    # S: base-centered
    # I: body-centered
    # F: face-centered
    #  - primitive triclinic: aP
    #  - primitive monoclinic: mP & mS
    #  - orthorhombic: oP, oS, oI, & oF
    #  - tetragonal: tP, tI
    #  - hexagonal: hR (rhombohedral) and hP (hexagonal)
    #  - cubic: cP, cI & cF
    reference = {
        '230': ['cI', 'cubic', 'cubic', 'cubic'],
        '229': ['cI', 'cubic', 'cubic', 'cubic'],
        '228': ['cF', 'cubic', 'cubic', 'cubic'],
        '227': ['cF', 'cubic', 'cubic', 'cubic'],
        '226': ['cF', 'cubic', 'cubic', 'cubic'],
        '225': ['cF', 'cubic', 'cubic', 'cubic'],
        '224': ['cP', 'cubic', 'cubic', 'cubic'],
        '223': ['cP', 'cubic', 'cubic', 'cubic'],
        '222': ['cP', 'cubic', 'cubic', 'cubic'],
        '221': ['cP', 'cubic', 'cubic', 'cubic'],
        '220': ['cI', 'cubic', 'cubic', 'cubic'],
        '219': ['cF', 'cubic', 'cubic', 'cubic'],
        '218': ['cP', 'cubic', 'cubic', 'cubic'],
        '217': ['cI', 'cubic', 'cubic', 'cubic'],
        '216': ['cF', 'cubic', 'cubic', 'cubic'],
        '215': ['cP', 'cubic', 'cubic', 'cubic'],
        '214': ['cI', 'cubic', 'cubic', 'cubic'],
        '213': ['cP', 'cubic', 'cubic', 'cubic'],
        '212': ['cP', 'cubic', 'cubic', 'cubic'],
        '211': ['cI', 'cubic', 'cubic', 'cubic'],
        '210': ['cF', 'cubic', 'cubic', 'cubic'],
        '209': ['cF', 'cubic', 'cubic', 'cubic'],
        '208': ['cP', 'cubic', 'cubic', 'cubic'],
        '207': ['cP', 'cubic', 'cubic', 'cubic'],
        '206': ['cI', 'cubic', 'cubic', 'cubic'],
        '205': ['cP', 'cubic', 'cubic', 'cubic'],
        '204': ['cI', 'cubic', 'cubic', 'cubic'],
        '203': ['cF', 'cubic', 'cubic', 'cubic'],
        '202': ['cF', 'cubic', 'cubic', 'cubic'],
        '201': ['cP', 'cubic', 'cubic', 'cubic'],
        '200': ['cP', 'cubic', 'cubic', 'cubic'],
        '199': ['cI', 'cubic', 'cubic', 'cubic'],
        '198': ['cP', 'cubic', 'cubic', 'cubic'],
        '197': ['cI', 'cubic', 'cubic', 'cubic'],
        '196': ['cF', 'cubic', 'cubic', 'cubic'],
        '195': ['cP', 'cubic', 'cubic', 'cubic'],
        '194': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '193': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '192': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '191': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '190': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '189': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '188': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '187': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '186': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '185': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '184': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '183': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '182': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '181': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '180': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '179': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '178': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '177': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '176': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '175': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '174': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '173': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '172': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '171': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '170': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '169': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '168': ['hP', 'hexagonal', 'hexagonal', 'hexagonal'],
        '167': ['hR', 'hexagonal', 'trigonal', 'rhombohedral'],
        '166': ['hR', 'hexagonal', 'trigonal', 'rhombohedral'],
        '165': ['hP', 'hexagonal', 'trigonal', 'hexagonal'],
        '164': ['hP', 'hexagonal', 'trigonal', 'hexagonal'],
        '163': ['hP', 'hexagonal', 'trigonal', 'hexagonal'],
        '162': ['hP', 'hexagonal', 'trigonal', 'hexagonal'],
        '161': ['hR', 'hexagonal', 'trigonal', 'rhombohedral'],
        '160': ['hR', 'hexagonal', 'trigonal', 'rhombohedral'],
        '159': ['hP', 'hexagonal', 'trigonal', 'hexagonal'],
        '158': ['hP', 'hexagonal', 'trigonal', 'hexagonal'],
        '157': ['hP', 'hexagonal', 'trigonal', 'hexagonal'],
        '156': ['hP', 'hexagonal', 'trigonal', 'hexagonal'],
        '155': ['hR', 'hexagonal', 'trigonal', 'rhombohedral'],
        '154': ['hP', 'hexagonal', 'trigonal', 'hexagonal'],
        '153': ['hP', 'hexagonal', 'trigonal', 'hexagonal'],
        '152': ['hP', 'hexagonal', 'trigonal', 'hexagonal'],
        '151': ['hP', 'hexagonal', 'trigonal', 'hexagonal'],
        '150': ['hP', 'hexagonal', 'trigonal', 'hexagonal'],
        '149': ['hP', 'hexagonal', 'trigonal', 'hexagonal'],
        '148': ['hR', 'hexagonal', 'trigonal', 'rhombohedral'],
        '147': ['hP', 'hexagonal', 'trigonal', 'hexagonal'],
        '146': ['hR', 'hexagonal', 'trigonal', 'rhombohedral'],
        '145': ['hP', 'hexagonal', 'trigonal', 'hexagonal'],
        '144': ['hP', 'hexagonal', 'trigonal', 'hexagonal'],
        '143': ['hP', 'hexagonal', 'trigonal', 'hexagonal'],
        '142': ['tI', 'tetragonal', 'tetragonal', 'tetragonal'],
        '141': ['tI', 'tetragonal', 'tetragonal', 'tetragonal'],
        '140': ['tI', 'tetragonal', 'tetragonal', 'tetragonal'],
        '139': ['tI', 'tetragonal', 'tetragonal', 'tetragonal'],
        '138': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '137': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '136': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '135': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '134': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '133': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '132': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '131': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '130': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '129': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '128': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '127': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '126': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '125': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '124': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '123': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '122': ['tI', 'tetragonal', 'tetragonal', 'tetragonal'],
        '121': ['tI', 'tetragonal', 'tetragonal', 'tetragonal'],
        '120': ['tI', 'tetragonal', 'tetragonal', 'tetragonal'],
        '119': ['tI', 'tetragonal', 'tetragonal', 'tetragonal'],
        '118': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '117': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '116': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '115': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '114': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '113': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '112': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '111': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '110': ['tI', 'tetragonal', 'tetragonal', 'tetragonal'],
        '109': ['tI', 'tetragonal', 'tetragonal', 'tetragonal'],
        '108': ['tI', 'tetragonal', 'tetragonal', 'tetragonal'],
        '107': ['tI', 'tetragonal', 'tetragonal', 'tetragonal'],
        '106': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '105': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '104': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '103': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '102': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '101': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '100': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '99': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '98': ['tI', 'tetragonal', 'tetragonal', 'tetragonal'],
        '97': ['tI', 'tetragonal', 'tetragonal', 'tetragonal'],
        '96': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '95': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '94': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '93': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '92': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '91': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '90': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '89': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '88': ['tI', 'tetragonal', 'tetragonal', 'tetragonal'],
        '87': ['tI', 'tetragonal', 'tetragonal', 'tetragonal'],
        '86': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '85': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '84': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '83': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '82': ['tI', 'tetragonal', 'tetragonal', 'tetragonal'],
        '81': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '80': ['tI', 'tetragonal', 'tetragonal', 'tetragonal'],
        '79': ['tI', 'tetragonal', 'tetragonal', 'tetragonal'],
        '78': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '77': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '76': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '75': ['tP', 'tetragonal', 'tetragonal', 'tetragonal'],
        '74': ['oI', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '73': ['oI', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '72': ['oI', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '71': ['oI', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '70': ['oF', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '69': ['oF', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '68': ['oC', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '67': ['oC', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '66': ['oC', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '65': ['oC', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '64': ['oC', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '63': ['oC', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '62': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '61': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '60': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '59': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '58': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '57': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '56': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '55': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '54': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '53': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '52': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '51': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '50': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '49': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '48': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '47': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '46': ['oI', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '45': ['oI', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '44': ['oI', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '43': ['oF', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '42': ['oF', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '41': ['oC', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '40': ['oC', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '39': ['oC', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '38': ['oC', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '37': ['oC', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '36': ['oC', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '35': ['oC', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '34': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '33': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '32': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '31': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '30': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '29': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '28': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '27': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '26': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '25': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '24': ['oI', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '23': ['oI', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '22': ['oF', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '21': ['oC', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '20': ['oC', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '19': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '18': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '17': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '16': ['oP', 'orthorhombic', 'orthorhombic', 'orthorhombic'],
        '15': ['mC', 'monoclinic', 'monoclinic', 'monoclinic'],
        '14': ['mP', 'monoclinic', 'monoclinic', 'monoclinic'],
        '13': ['mP', 'monoclinic', 'monoclinic', 'monoclinic'],
        '12': ['mC', 'monoclinic', 'monoclinic', 'monoclinic'],
        '11': ['mP', 'monoclinic', 'monoclinic', 'monoclinic'],
        '10': ['mP', 'monoclinic', 'monoclinic', 'monoclinic'],
        '9': ['mC', 'monoclinic', 'monoclinic', 'monoclinic'],
        '8': ['mC', 'monoclinic', 'monoclinic', 'monoclinic'],
        '7': ['mP', 'monoclinic', 'monoclinic', 'monoclinic'],
        '6': ['mP', 'monoclinic', 'monoclinic', 'monoclinic'],
        '5': ['mC', 'monoclinic', 'monoclinic', 'monoclinic'],
        '4': ['mP', 'monoclinic', 'monoclinic', 'monoclinic'],
        '3': ['mP', 'monoclinic', 'monoclinic', 'monoclinic'],
        '2': ['aP', 'triclinic', 'triclinic', 'triclinic'],
        '1': ['aP', 'triclinic', 'triclinic', 'triclinic'],
        }
    return reference[str(space_group)]

def verify_crystal_system_bravais_lattice_consistency(crystal_system, bravais_lattice):
    # Crystal system is reported in the csd entry
    # Bravais lattice is deduced from the reported space group number
    check = False
    if crystal_system == 'cubic':
        if bravais_lattice in ['cF', 'cI', 'cP']:
            check = True
    elif crystal_system == 'monoclinic':
        if bravais_lattice in ['mC', 'mP']:
            check = True
    elif crystal_system == 'orthorhombic':
        if bravais_lattice in ['oC', 'oF', 'oI', 'oP']:
            check = True
    elif crystal_system == 'tetragonal':
        if bravais_lattice in ['tI', 'tP']:
            check = True
    elif crystal_system == 'triclinic':
        if bravais_lattice == 'aP':
            check = True
    elif crystal_system == 'hexagonal':
        if bravais_lattice in ['hR', 'hP']:
            check = True
    elif crystal_system == 'trigonal':
        if bravais_lattice in ['hR', 'hP']:
            check = True
    elif crystal_system == 'rhombohedral':
        if bravais_lattice in ['hR', 'hP']:
            check = True
    return check

def verify_unit_cell_consistency(crystal_system, bravais_lattice, cell_lengths, cell_angles):
    def square(alpha, beta, gamma):
        if alpha == 90 and beta == 90 and gamma == 90:
            return True
        else:
            return False

    check = False
    a, b, c = cell_lengths
    alpha, beta, gamma = cell_angles

    if bravais_lattice in ['cF', 'cI', 'cP']:
        if square(alpha, beta, gamma):
            if a == b and b == c:
                check = True
    elif bravais_lattice in ['tI', 'tP']:
        if square(alpha, beta, gamma):
            if a == b or a == c or b == c:
                check = True
    elif bravais_lattice in ['oC', 'oF', 'oI', 'oP']:
        if square(alpha, beta, gamma):
            check = True
    elif bravais_lattice in ['hR', 'hP']:
        # It seems like there are lots of mislabeld hR as hP and visa-versa. Vanessa had
        # lumped these two together, so I am doing the same
        if a == b and b == c:
            if alpha == beta and beta == gamma:
                check = True
        if a == b or b == c or a == c:
            if alpha == 90 and beta == 90 and gamma == 120:
                check = True
            elif alpha == 90 and beta == 120 and gamma == 90:
                check = True
            elif alpha == 120 and beta == 90 and gamma == 90:
                check = True
    elif bravais_lattice in ['mC', 'mP']:
        if alpha == beta and alpha == 90 and gamma != 90:
            check = True
        elif alpha == gamma and alpha == 90 and beta != 90:
            check = True
        elif beta == gamma and beta == 90 and alpha != 90:
            check = True
    elif bravais_lattice == 'aP':
        check = True
    return check

def verify_volume(cell_lengths, cell_angles, expected_volume):
    a = cell_lengths[0]
    b = cell_lengths[1]
    c = cell_lengths[2]
    calpha = np.cos(np.pi/180 * cell_angles[0])
    cbeta = np.cos(np.pi/180 * cell_angles[1])
    cgamma = np.cos(np.pi/180 * cell_angles[2])
    volume = a*b*c * np.sqrt(1 - calpha**2 - cbeta**2 - cgamma**2 + 2*calpha*cbeta*cgamma)
    return np.isclose(volume, expected_volume)

class ChemicalFormulaHandler:
    def __init__(self, csd_formula):
        self.csd_formula = csd_formula
        self.status = True

    def get_chemical_composition_dict(self):
        # https://www.iucr.org/__data/iucr/cifdic_html/1/cif_core.dic/Ichemical_formula_moiety.html
        # https://www.iucr.org/__data/iucr/cifdic_html/1/cif_core.dic/Cchemical_formula.html
        # https://www.iucr.org/__data/iucr/cifdic_html/1/cif_core.dic/Ichemical_formula_sum.html
        if ',' in self.csd_formula:
            # multiple moieties
            moieties = self.csd_formula.split(',')
            chemical_dicts = [i for i in range(len(moieties))]
            for m_index, moiety in enumerate(moieties):
                moiety = moiety.strip()
                chemical_dicts[m_index] = process_moiety(moiety)
            self.chemical_dict = combine_chemical_dicts(chemical_dicts)
        else:
            # single moiety
            # Two forms: AbX CdY and (AbX CdY)n
            self.chemical_dict = process_moiety(self.csd_formula)

    def count_non_hydrogen_atoms(self):
        self.n_non_hydrogen_atoms = np.sum(list(self.chemical_dict.values()))

    def get_chemical_composition_string(self):
        '''
        This converts the chemical composition dictionary to a string that can be used to 
        identify duplicates.
            - The rounding to the nearest 0 decimal place will convert something with an atom
              position that has two atoms with 0.01 and 0.99 to what would be equivalent for
              just the most common atom.
            - I am dividing by 2 so multiples can be reduced
        '''
        keys = []
        values = []
        for key in self.chemical_dict.keys():
            value = int(np.round(self.chemical_dict[key], decimals=0))
            if value > 0:
                keys.append(key)
                values.append(value)
        if len(keys) == 0:
            return ''
        else:
            divisible = True
            index = 0
            while divisible:
                if all([x % 2 == 0 for x in values]):
                    values = [x // 2 for x in values]
                    index += 1
                else:
                    divisible = False
                if index > 10:
                    print(f'hung {keys} {values}')
            self.chemical_string = ' '.join(
                [f'{keys[i]}{values[i]}' for i in range(len(keys))]
                )

def get_empty_chemical_dict():
    symbols = [
        'H','He','Li','Be','B','C','N','O','F','Ne',
        'Na','Mg','Al','Si','P','S','Cl','Ar','K', 'Ca',
        'Sc', 'Ti', 'V','Cr', 'Mn', 'Fe', 'Co', 'Ni',
        'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
        'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
        'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
        'I', 'Xe','Cs', 'Ba','La', 'Ce', 'Pr', 'Nd', 'Pm',
        'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
        'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
        'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
        'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am',
        'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
        'Rf', 'Db', 'Sg', 'Bh','Hs', 'Mt', 'Ds', 'Rg', 'Cn',
        'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
        ]
    symbols.sort()
    chemical_dict = dict.fromkeys(symbols, 0)
    return chemical_dict

def get_atom_types(pairs):
    chemical_dict = get_empty_chemical_dict()
    for pair in pairs:
        if pair[0] in chemical_dict.keys():
            chemical_dict[pair[0]] = pair[1]
    return chemical_dict

def simple_formula(chemical_formula, multiplier=1):
    pairs = []
    for element in chemical_formula.split(' '):
        # An element like 1- or 3+ indicates charge
        if not any([x in element for x in ['+', '-']]):
            for str_index, char in enumerate(element):
                if char.isdigit():
                    atom_name = element[:str_index]
                    # Replace deuteriums with hydrogen
                    if atom_name == 'D':
                        atom_name = 'H'
                    number = multiplier * float(element[str_index:])
                    pairs.append([atom_name, number])
                    break
                elif str_index == len(element) - 1:
                    atom_name = element
                    # Replace deuteriums with hydrogen
                    if atom_name == 'D':
                        atom_name = 'H'
                    number = multiplier
                    pairs.append([atom_name, number])
    chemical_dict = get_atom_types(pairs)    
    return chemical_dict

def process_moiety(moiety):
    multiplier = 1
    reduced = moiety
    if moiety.endswith(')n'):
        # Chemical formula is (Ax By)n
        if moiety[0].isdigit():
            multiplier = float(moiety.split('(')[0])
            reduced = moiety.replace(')n', '').split('(')[1]
        else:
            reduced = moiety.replace(')n', '').replace('(', '')
    elif moiety.endswith('n') and moiety[-2].isdigit():
        multiplier = float(moiety.split(')')[1].replace('n', ''))
        reduced = moiety.split('(')[1].split(')')[0]
    elif moiety.startswith('n') and moiety[1].isdigit():
        multiplier = float(moiety.split('(')[0].replace('n', ''))
        reduced = moiety.split('(')[1].split(')')[0]
    elif moiety.startswith('n('):
        # Chemical formula is n(Ax By)
        reduced = moiety.replace('n(', '').replace(')', '')
    elif 'n(' in moiety:
        multiplier = float(moiety.split('n(')[0])
        reduced = moiety.split('n(')[1].replace(')', '')
    elif moiety.startswith('x('):
        # Chemical formula is x(Ax By)
        reduced = moiety.replace('x(', '').replace(')', '')
    elif 'x(' in moiety:
        multiplier = float(moiety.split('x(')[0])
        reduced = moiety.split('x(')[1].replace(')', '')
    elif moiety.startswith('y('):
        # Chemical formula is y(Ax By)
        reduced = moiety.replace('y(', '').replace(')', '')
    elif 'y(' in moiety:
        multiplier = float(moiety.split('y(')[0])
        reduced = moiety.split('y(')[1].replace(')', '')
    elif moiety.startswith('z('):
        # Chemical formula is z(Ax By)
        reduced = moiety.replace('z(', '').replace(')', '')
    elif 'z(' in moiety:
        multiplier = float(moiety.split('z(')[0])
        reduced = moiety.split('z(')[1].replace(')', '')
    elif moiety[0].isdigit():
        multiplier = float(moiety.split('(')[0])
        reduced = moiety.split('(')[1].split(')')[0]
    return simple_formula(reduced, multiplier)

def combine_chemical_dicts(chemical_dicts):
    chemical_dict = get_empty_chemical_dict()
    for d in chemical_dicts:
        for key in d.keys():
            chemical_dict[key] += d[key]
    return chemical_dict


def calculate_g6(reduced_uc_params):
    mm11 = np.square(reduced_uc_params[0])
    mm22 = np.square(reduced_uc_params[1])
    mm33 = np.square(reduced_uc_params[2])
    mm12 = 2*reduced_uc_params[1]*reduced_uc_params[2]*np.cos(np.radians(reduced_uc_params[3]))
    mm23 = 2*reduced_uc_params[0]*reduced_uc_params[2]*np.cos(np.radians(reduced_uc_params[4]))
    mm31 = 2*reduced_uc_params[0]*reduced_uc_params[1]*np.cos(np.radians(reduced_uc_params[5]))
    return np.array([mm11, mm22, mm33, mm12, mm23, mm31])

def calculate_s6(g6_params):
    #a = reduced_uc_params[0]
    #b = reduced_uc_params[1]
    #c = reduced_uc_params[2]
    #d = -1*(a+b+c)
    # [P, Q, R, S, T, U] = [s23, s13, s12, s14, s24, s34] = [bdotc, adotc, adotb, adotd, bdotd, cdotd]
    tol = 1e-10
    p = g6_params[3]/2.0
    q = g6_params[4]/2.0
    r = g6_params[5]/2.0
    s = (-2.0*g6_params[0] - g6_params[5] - g6_params[4]) / 2.0
    t = (-1.0*g6_params[5] - 2.0*g6_params[1] - g6_params[3]) / 2.0
    u = (-1.0*g6_params[4] - g6_params[3] - 2.0*g6_params[2]) / 2.0
    #tselling_vector = np.array([np.dot(b,c), np.dot(a,c), np.dot(a,b),
    #                           np.dot(a,d), np.dot(b,d), np.dot(c,d),
    #                          ]
    #                             )
    selling_vector = np.array([p, q, r, s, t, u])
    selling_vector = np.array([s if abs(s) > tol else 0 for s in selling_vector])
    #print(tselling_vector,selling_vector)
    reduction_matrices = np.array(
                                      [
                                       np.array(
                                                [
                                                 [-1, 0, 0, 0, 0, 0],
                                                 [ 1, 1, 0, 0, 0, 0],
                                                 [ 1, 0, 0, 0, 1, 0],
                                                 [-1, 0, 0, 1, 0, 0],
                                                 [ 1, 0, 1, 0, 0, 0],
                                                 [ 1, 0, 0, 0, 0, 1],
                                                ]
                                               ),
                                       np.array(
                                                [
                                                 [ 1, 1, 0, 0, 0, 0],
                                                 [ 0,-1, 0, 0, 0, 0],
                                                 [ 0, 1, 0, 1, 0, 0],
                                                 [ 0, 1, 1, 0, 0, 0],
                                                 [ 0,-1, 0, 0, 1, 0],
                                                 [ 0, 1, 0, 0, 0, 1],
                                                ]
                                               ),
                                       np.array(
                                                [
                                                 [ 1, 0, 1, 0, 0, 0],
                                                 [ 0, 0, 1, 1, 0, 0],
                                                 [ 0, 0,-1, 0, 0, 0],
                                                 [ 0, 1, 1, 0, 0, 0],
                                                 [ 0, 0, 1, 0, 1, 0],
                                                 [ 0, 0,-1, 0, 0, 1],
                                                ]
                                               ),
                                       np.array(
                                                [
                                                 [ 1, 0, 0,-1, 0, 0],
                                                 [ 0, 0, 1, 1, 0, 0],
                                                 [ 0, 1, 0, 1, 0, 0],
                                                 [ 0, 0, 0,-1, 0, 0],
                                                 [ 0, 0, 0, 1, 1, 0],
                                                 [ 0, 0, 0, 1, 0, 1],
                                                ]
                                               ),
                                     np.array(
                                              [
                                               [ 0, 0, 1, 0, 1, 0],
                                               [ 0, 1, 0, 0,-1, 0],
                                               [ 1, 0, 0, 0, 1, 0],
                                               [ 0, 0, 0, 1, 1, 0],
                                               [ 0, 0, 0, 0,-1, 0],
                                               [ 0, 0, 0, 0, 1, 1],
                                              ]
                                             ),
                                     np.array(
                                              [
                                               [ 0, 1, 0, 0, 0, 1],
                                               [ 1, 0, 0, 0, 0, 1],
                                               [ 0, 0, 1, 0, 0,-1],
                                               [ 0, 0, 0, 1, 0, 1],
                                               [ 0, 0, 0, 0, 1, 1],
                                               [ 0, 0, 0, 0, 0,-1],
                                              ]
                                             ),
                                  ]
                                 )

    while np.greater(np.max(selling_vector), 0):
        max_index = selling_vector.argmax()
        selling_vector = np.dot(reduction_matrices[max_index], selling_vector)
    #selling_vector = np.array([s if abs(s) > tol else 0 for s in selling_vector])

    if np.max(selling_vector) > 0:
        print("Selling vector reduction failed")    
        return None
    else:
        return selling_vector

def selling_distance(svector1, svector2):
    vcp_transform_mats = [
         np.array(
                  [
                       [-1, 0, 0, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                      ]
                     ),
             np.array(
                      [
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0,-1, 0, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 0, 0, 1],
                      ]
                     ),
             np.array(
                      [
                       [ 1, 0, 0, 0, 0, 0,],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0,-1, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 0, 0, 1],
                      ]
                     ),
             np.array(
                      [
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0,-1, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 0, 0, 1],
                      ]
                     ),
             np.array(
                      [
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0,-1, 0],
                       [ 0, 0, 0, 0, 0, 1],
                      ]
                     ), 
             np.array(
                      [
                       [ 0, 1, 0, 0, 0, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0 ,0 ,0 ,0, 0,-1],
                  ]
                 ),
        ] 


    reflection_mats = [
         np.array(
                      [
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 0, 0, 1],
                      ]
                     ),
             np.array(
                      [
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 0, 0, 0, 1, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0 ,0 ,0, 0, 1, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0 ,0 ,0, 0, 0, 1],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 1, 0, 0, 0, 0],
                       [ 1 ,0, 0, 0, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 0, 0, 1, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                      ]
                     ),
                        
             np.array(
                      [ 
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 1, 0, 0, 0, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 1, 0, 0, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                      ]
                     ),
             np.array( #10
                      [ 
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 1, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 1, 0, 0, 0, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 1, 0, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 1, 0, 0, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 0, 0, 1, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 0, 0, 1, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 1, 0, 0, 0, 0, 0],
                      ]
                     ),
             np.array(#20
                      [ 
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 1],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 0, 0, 0, 1],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 0, 0, 0, 1],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 0, 0, 1, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 1, 0],
                      ]
                     ),
             np.array(
                      [ 
                       [ 0, 0, 0, 0, 0, 1],
                       [ 0, 0, 0, 0, 1, 0],
                       [ 1, 0, 0, 0, 0, 0],
                       [ 0, 0, 1, 0, 0, 0],
                       [ 0, 1, 0, 0, 0, 0],
                       [ 0, 0, 0, 1, 0, 0],
                      ]
                 ),
        ]

    vcps = np.dot(svector1, vcp_transform_mats)[0]

    all_reflections = []
    for vcp in vcps:
        for mat in reflection_mats:
            all_reflections.append(np.dot(vcp, mat))
    for mat in reflection_mats:
        all_reflections.append(np.dot(svector1, reflection_mat))
         
    return np.min(np.linalg.norm(reflection-svector2) for reflection in all_reflections)

def calculate_uc_from_g6(g6_params):
    g1 = g6_params[0]
    g2 = g6_params[1]
    g3 = g6_params[2]
    g4 = g6_params[3]
    g5 = g6_params[4]
    g6 = g6_params[5]
    a = np.sqrt(np.abs(g1))
    b = np.sqrt(np.abs(g2))
    c = np.sqrt(np.abs(g3))
    alpha = (np.arccos(g4/(2*b*c)))*(180.0/np.pi)
    beta = (np.arccos(g5/(2*a*c)))*(180.0/np.pi)
    gamma = (np.arccos(g6/(2*a*b)))*(180.0/np.pi)
    return [a, b, c, alpha, beta, gamma]

def calculate_uc_from_s6(s6_params):
    p = s6_params[0]
    q = s6_params[1]
    r = s6_params[2]
    s = s6_params[3]
    t = s6_params[4]
    u = s6_params[5]
    g1 = -1*q-r-s
    g2 = -1*p-r-t
    g3 = -1*p-q-u
    g4 = 2*p
    g5 = 2*q
    g6 = 2*r
    return calculate_uc_from_g6([g1,g2,g3,g4,g5,g6])