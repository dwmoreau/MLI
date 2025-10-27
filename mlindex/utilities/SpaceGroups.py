import numpy as np


def get_spacegroup_hkl_ref(hkl_ref, bravais_lattice):
    import gemmi

    # https://www.ba.ic.cnr.it/softwareic/expo/extinction_symbols/
    if bravais_lattice == "cF":
        spacegroups = ["F 2 3", "F d -3", "F 41 3 2", "F -4 3 c", "F d -3 c"]
        extinction_groups = ["F - - -", "F d - -", "F 41 - -", "F - - c", "F d - c"]
    elif bravais_lattice == "cI":
        spacegroups = ["I 2 3", "I a -3", "I 41 3 2", "I -4 3 d", "I a -3 d"]
        extinction_groups = ["I - - -", "I a - -", "I 41 - -", "I - - d", "I a - d"]
    elif bravais_lattice == "cP":
        spacegroups = [
            "P 2 3",
            "P 21 3",
            "P n -3",
            "P a -3",
            "P 43 3 2",
            "P -43 n",
            "P n -3 n",
        ]
        extinction_groups = [
            "P - - -",
            "P 21 - -",
            "P n - -",
            "P a - -",
            "P 41 - -",
            "P - - n",
            "P n - n",
        ]
    elif bravais_lattice == "hR":
        spacegroups = ["R 3", "R 3 c"]
        extinction_groups = ["R - - -", "R - - c"]
    elif bravais_lattice == "hP":
        spacegroups = [
            "P 3",
            "P 31",
            "P 3 c 1",
            "P 3 1 c",
            "P 61",
            "P 62",
            "P 63",
            "P 6 c c",
        ]
        extinction_groups = [
            "P - - -",
            "P 31 - -",
            "P - c -",
            "P - - c",
            "P 61 - -",
            "P 62 - -",
            "P 63 - -",
            "P - c c",
        ]
    elif bravais_lattice == "tI":
        spacegroups = [
            "I 4",
            "I 41",
            "I 41/a",
            "I 4 c m",
            "I 41 m d",
            "I 41 c d",
            "I 41/a m d",
            "I 41/a c d",
        ]
        extinction_groups = [
            "I - - -",
            "I41 - -",
            "I 41/a - -",
            "I - c -",
            "I - - d",
            "I - c d",
            "I a - d",
            "I a c d",
        ]
    elif bravais_lattice == "tP":
        spacegroups = [
            "P 4",
            "P 41",
            "P 42",
            "P 4/n",
            "P 42/n",
            "P 4 21 2",
            "P 41 21 2",
            "P 42 21 2",
            "P 4 b m",
            "P 42 c m",
            "P 42 n m",
            "P 4 c c",
            "P 4 n c",
            "P 42 m c",
            "P 42 b c",
            "P -4 21 c",
            "P 4/n b m",
            "P 4/n n c",
            "P 4/n c c",
            "P 42/n b c",
            "P 42/n n m",
            "P 42/n m c",
            "P 42/n c m",
        ]
        extinction_groups = [
            "P - - -",
            "P 41 - -",
            "P 42 - -",
            "P n - -",
            "P 42/n - -",
            "P - 21 2",
            "P 41 21 -",
            "P 42 21 2",
            "P - b -",
            "P - c -",
            "P - n -",
            "P - c c",
            "P - n c",
            "P - - c",
            "P - b c",
            "P - c1 c",
            "P n b -",
            "P n n c",
            "P n c c",
            "P n b c",
            "P n n -",
            "P n - c",
            "P n c -",
        ]
    elif bravais_lattice == "oC":
        # Do I need to add C m 2 b for example?
        spacegroups = [
            "C 2 2 2",
            "C 2 2 21",
            "C c c 2",
            "C c 2 m",
            "C 2 c m",
            "C c 2 a",
            "C 2 c b",
            "C c c a",
            "C m 2 a",
        ]
        extinction_groups = [
            "C - - -",
            "C - - 21",
            "C c c -",
            "C c - -",
            "C - c -",
            "C c - a",
            "C - c b",
            "C c c a",
            "C - - a",
        ]
    elif bravais_lattice == "oF":
        spacegroups = ["F 2 2 2", "F d d d", "F 2 d d", "F d 2 d", "F d d 2"]
        extinction_groups = ["F - - -", "F d d d", "F - d d", "F d - d", "F d d -"]
    elif bravais_lattice == "oI":
        spacegroups = [
            "I m m m",
            "I b c a",
            "I b a 2",
            "I 2 c b",
            "I c 2 a",
            "I b m 2",
            "I m a 2",
            "I m 2 a",
        ]
        extinction_groups = [
            "I - - -",
            "I b c a",
            "I b a -",
            "I - c b",
            "I c - a",
            "I b m -",
            "I - a -",
            "I - - a",
        ]
    elif bravais_lattice == "oP":
        spacegroups = [
            "P 2 2 2",
            "P 21 2 2",
            "P 2 21 2",
            "P 2 2 21",
            "P 21 m a",
            "P m 21 b",
            "P m c 21",
            "P 21 a m",
            "P b 21 m",
            "P c m 21",
            "P 2 a a",
            "P b 2 b",
            "P c c 2",
            "P 2 21 21",
            "P 21 2 21",
            "P 21 21 2",
            "P b c 21",
            "P c a 21",
            "P b 21 a",
            "P c 21 b",
            "P 21 c a",
            "P 21 a b",
            "P 2 c b",
            "P c 2 a",
            "P b a 2",
            "P n c 2",
            "P c n 2",
            "P b 2 n",
            "P n 2 b",
            "P 2 n a",
            "P 2 a n",
            "P n m 21",
            "P m 21 n",
            "P 21 n m",
            "P n c b",
            "P c n a",
            "P b a n",
            "P c c b",
            "P c c a",
            "P b a a",
            "P b c b",
            "P c a a",
            "P b a b",
            "P 21 21 21",
            "P n n n",
            "P b c a",
            "P c a b",
            "P n a 21",
            "P b n 21",
            "P c 21 n",
            "P n 21 a",
            "P 21 n b",
            "P 21 c n",
            "P 2 n n",
            "P n 2 n",
            "P n n 2",
            "P n n a",
            "P n n b",
            "P n c n",
            "P c c n",
            "P b n b",
            "P n a a",
            "P b c n",
            "P c a n",
            "P b n a",
            "P c n b",
            "P n c a",
            "P n a b",
        ]
        extinction_groups = [
            "P - - -",
            "P 21 - -",
            "P - 21 -",
            "P - - 21",
            "P - - a",
            "P - - b",
            "P - c -",
            "P - a -",
            "P b - -",
            "P c - -",
            "P - a a",
            "P b - b",
            "P c c -",
            "P - 21 21",
            "P 21 - 21",
            "P 21 21 -",
            "P b c -",
            "P c a -",
            "P b - a",
            "P c - b",
            "P - c a",
            "P - a b",
            "P - c b",
            "P c - a",
            "P b a -",
            "P n c -",
            "P c n -",
            "P b - n",
            "P n - b",
            "P - n a",
            "P - a n",
            "P n - -",
            "P - - n",
            "P - n -",
            "P n c b",
            "P c n a",
            "P b a n",
            "P c c b",
            "P c c a",
            "P b a a",
            "P b c b",
            "P c a a",
            "P b a b",
            "P 21 21 21",
            "P n n n",
            "P b c a",
            "P c a b",
            "P n a -",
            "P b n -",
            "P c - n",
            "P n - a",
            "P - n b",
            "P - c n",
            "P - n n",
            "P n - n",
            "P n n -",
            "P n n a",
            "P n n b",
            "P n c n",
            "P c c n",
            "P b n b",
            "P n a a",
            "P b c n",
            "P c a n",
            "P b n a",
            "P c n b",
            "P n c a",
            "P n a b",
        ]
    elif bravais_lattice == "mC":
        spacegroups = ["I 1 2 1", "I 1 a 1"]
        extinction_groups = ["I 1 - 1", "I 1 a 1"]
    elif bravais_lattice == "mP":
        spacegroups = [
            "P 1 2 1",
            "P 1 21 1",
            "P 1 c 1",
            "P 1 a 1",
            "P 1 n 1",
            "P 1 21/c 1",
            "P 1 21/a 1",
            "P 1 21/n 1",
        ]
        extinction_groups = [
            "P 1 - 1",
            "P 1 21 1",
            "P 1 c 1",
            "P 1 a 1",
            "P 1 n 1",
            "P 1 21/c 1",
            "P 1 21/a 1",
            "P 1 21/n 1",
        ]
    elif bravais_lattice == "aP":
        spacegroups = ["P 1"]
        extinction_groups = ["P -"]
    """
    hkl_ref_sg = dict.fromkeys(spacegroups)
    for spacegroup in spacegroups:
        if bravais_lattice == 'hR':
            # gemmi gives the systematic absences for rhombohedral in the hexagonal setting.
            # The ':R' component tells gemmi to use the rhombohedral setting
            ops = gemmi.SpaceGroup(f'{spacegroup}:R').operations()
        else:
            ops = gemmi.SpaceGroup(spacegroup).operations()
        systematically_absent = ops.systematic_absences(hkl_ref)
        hkl_ref_sg[spacegroup] = hkl_ref[np.invert(systematically_absent)]
    """
    keys = [f"{i} e.g. {j}" for i, j in zip(extinction_groups, spacegroups)]
    hkl_ref_sg = dict.fromkeys(keys)
    for index, key in enumerate(keys):
        if bravais_lattice == "hR":
            # gemmi gives the systematic absences for rhombohedral in the hexagonal setting.
            # The ':R' component tells gemmi to use the rhombohedral setting
            ops = gemmi.SpaceGroup(f"{spacegroups[index]}:R").operations()
        else:
            ops = gemmi.SpaceGroup(spacegroups[index]).operations()
        systematically_absent = ops.systematic_absences(hkl_ref)
        hkl_ref_sg[key] = hkl_ref[np.invert(systematically_absent)]
    return hkl_ref_sg


def map_spacegroup_to_extinction_group(spacegroup_symbol_hm):
    # These extinction groups are based on those used by EXPO
    #   https://www.ba.ic.cnr.it/softwareic/expo/extinction_symbols/
    # This function was mostly produced by ChatGPT.
    table_data = [
        {
            "Code": 15,
            "Extinction Group": "P 1 – 1",
            "Space Groups": ["P121", "P1m1", "P12/m1"],
        },
        {
            "Code": 16,
            "Extinction Group": "P 1 21 1",
            "Space Groups": ["P1211", "P121/m1"],
        },
        {"Code": 21, "Extinction Group": "P 1 n 1", "Space Groups": ["P1n1", "P12/n1"]},
        {"Code": 22, "Extinction Group": "P 1 21/n 1", "Space Groups": ["P121/n1"]},
        {
            "Code": 27,
            "Extinction Group": "I 1 - 1",
            "Space Groups": ["I121", "I1m1", "I12/m1"],
        },
        {"Code": 28, "Extinction Group": "I 1 a 1", "Space Groups": ["I1a1", "I12/a1"]},
        {
            "Code": 43,
            "Extinction Group": "P – – –",
            "Space Groups": ["P222", "Pmm2", "Pmmm", "Pm2m", "P2mm"],
        },
        {"Code": 44, "Extinction Group": "P – – 21", "Space Groups": ["P2221"]},
        {"Code": 45, "Extinction Group": "P – 21 –", "Space Groups": ["P2212"]},
        {"Code": 46, "Extinction Group": "P – 21 21", "Space Groups": ["P22121"]},
        {"Code": 47, "Extinction Group": "P 21 – –", "Space Groups": ["P2122"]},
        {"Code": 48, "Extinction Group": "P 21 – 21", "Space Groups": ["P21221"]},
        {"Code": 49, "Extinction Group": "P 21 21 –", "Space Groups": ["P21212"]},
        {"Code": 50, "Extinction Group": "P 21 21 21", "Space Groups": ["P212121"]},
        {
            "Code": 51,
            "Extinction Group": "P – – a",
            "Space Groups": ["Pm2a", "P21ma", "Pmma"],
        },
        {
            "Code": 52,
            "Extinction Group": "P – – b",
            "Space Groups": ["Pm21b", "P2mb", "Pmmb"],
        },
        {
            "Code": 53,
            "Extinction Group": "P – – n",
            "Space Groups": ["Pm21n", "P21mn", "Pmmn"],
        },
        {
            "Code": 54,
            "Extinction Group": "P – a –",
            "Space Groups": ["Pma2", "Pmam", "P21am"],
        },
        {"Code": 55, "Extinction Group": "P – a a", "Space Groups": ["P2aa", "Pmaa"]},
        {"Code": 56, "Extinction Group": "P – a b", "Space Groups": ["P21ab", "Pmab"]},
        {"Code": 57, "Extinction Group": "P – a n", "Space Groups": ["P2an", "Pman"]},
        {
            "Code": 58,
            "Extinction Group": "P – c –",
            "Space Groups": ["Pmc21", "P2cm", "Pmcm"],
        },
        {"Code": 59, "Extinction Group": "P – c a", "Space Groups": ["P21ca", "Pmca"]},
        {"Code": 60, "Extinction Group": "P – c b", "Space Groups": ["P2cb", "Pmcb"]},
        {"Code": 61, "Extinction Group": "P – c n", "Space Groups": ["P21cn", "Pmcn"]},
        {
            "Code": 62,
            "Extinction Group": "P – n –",
            "Space Groups": ["Pmn21", "P21nm", "Pmnm"],
        },
        {"Code": 63, "Extinction Group": "P – n a", "Space Groups": ["P2na", "Pmna"]},
        {"Code": 64, "Extinction Group": "P – n b", "Space Groups": ["P21nb", "Pmnb"]},
        {"Code": 65, "Extinction Group": "P – n n", "Space Groups": ["P2nn", "Pmnn"]},
        {
            "Code": 66,
            "Extinction Group": "P b – –",
            "Space Groups": ["Pbm2", "Pb21m", "Pbmm"],
        },
        {"Code": 67, "Extinction Group": "P b – a", "Space Groups": ["Pb21a", "Pbma"]},
        {"Code": 68, "Extinction Group": "P b – b", "Space Groups": ["Pb2b", "Pbmb"]},
        {"Code": 69, "Extinction Group": "P b – n", "Space Groups": ["Pb2n", "Pbmn"]},
        {"Code": 70, "Extinction Group": "P b a –", "Space Groups": ["Pba2", "Pbam"]},
        {"Code": 71, "Extinction Group": "P b a a", "Space Groups": ["Pbaa"]},
        {"Code": 72, "Extinction Group": "P b a b", "Space Groups": ["Pbab"]},
        {"Code": 73, "Extinction Group": "P b a n", "Space Groups": ["Pban"]},
        {"Code": 74, "Extinction Group": "P b c –", "Space Groups": ["Pbc21", "Pbcm"]},
        {"Code": 75, "Extinction Group": "P b c a", "Space Groups": ["Pbca"]},
        {"Code": 76, "Extinction Group": "P b c b", "Space Groups": ["Pbcb"]},
        {"Code": 77, "Extinction Group": "P b c n", "Space Groups": ["Pbcn"]},
        {"Code": 78, "Extinction Group": "P b n -", "Space Groups": ["Pbn21", "Pbnm"]},
        {"Code": 79, "Extinction Group": "P b n a", "Space Groups": ["Pbna"]},
        {"Code": 80, "Extinction Group": "P b n b", "Space Groups": ["Pbnb"]},
        {"Code": 81, "Extinction Group": "P b n n", "Space Groups": ["Pbnn"]},
        {
            "Code": 82,
            "Extinction Group": "P c – –",
            "Space Groups": ["Pcm21", "Pc2m", "Pcmm"],
        },
        {"Code": 83, "Extinction Group": "P c – a", "Space Groups": ["Pc2a", "Pcma"]},
        {"Code": 84, "Extinction Group": "P c – b", "Space Groups": ["Pc21b", "Pcmb"]},
        {"Code": 85, "Extinction Group": "P c – n", "Space Groups": ["Pc21n", "Pcmn"]},
        {"Code": 86, "Extinction Group": "P c a –", "Space Groups": ["Pca21", "Pcam"]},
        {"Code": 87, "Extinction Group": "P c a a", "Space Groups": ["Pcaa"]},
        {"Code": 88, "Extinction Group": "P c a b", "Space Groups": ["Pcab"]},
        {"Code": 89, "Extinction Group": "P c a n", "Space Groups": ["Pcan"]},
        {"Code": 90, "Extinction Group": "P c c –", "Space Groups": ["Pcc2", "Pccm"]},
        {"Code": 91, "Extinction Group": "P c c a", "Space Groups": ["Pcca"]},
        {"Code": 92, "Extinction Group": "P c c b", "Space Groups": ["Pccb"]},
        {"Code": 93, "Extinction Group": "P c c n", "Space Groups": ["Pccn"]},
        {"Code": 94, "Extinction Group": "P c n –", "Space Groups": ["Pcn2", "Pcnm"]},
        {"Code": 95, "Extinction Group": "P c n a", "Space Groups": ["Pcna"]},
        {"Code": 96, "Extinction Group": "P c n b", "Space Groups": ["Pcnb"]},
        {"Code": 97, "Extinction Group": "P c n n", "Space Groups": ["Pcnn"]},
        {
            "Code": 98,
            "Extinction Group": "P n – –",
            "Space Groups": ["Pnm21", "Pnmm", "Pn21m"],
        },
        {"Code": 99, "Extinction Group": "P n – a", "Space Groups": ["Pn21a", "Pnma"]},
        {"Code": 100, "Extinction Group": "P n – b", "Space Groups": ["Pn2b", "Pnmb"]},
        {"Code": 101, "Extinction Group": "P n – n", "Space Groups": ["Pn2n", "Pnmn"]},
        {"Code": 102, "Extinction Group": "P n a –", "Space Groups": ["Pna21", "Pnam"]},
        {"Code": 103, "Extinction Group": "P n a a", "Space Groups": ["Pnaa"]},
        {"Code": 104, "Extinction Group": "P n a b", "Space Groups": ["Pnab"]},
        {"Code": 105, "Extinction Group": "P n a n", "Space Groups": ["Pnan"]},
        {"Code": 106, "Extinction Group": "P n c –", "Space Groups": ["Pnc2", "Pncm"]},
        {"Code": 107, "Extinction Group": "P n c a", "Space Groups": ["Pnca"]},
        {"Code": 108, "Extinction Group": "P n c b", "Space Groups": ["Pncb"]},
        {"Code": 109, "Extinction Group": "P n c n", "Space Groups": ["Pncn"]},
        {"Code": 110, "Extinction Group": "P n n –", "Space Groups": ["Pnn2", "Pnnm"]},
        {"Code": 111, "Extinction Group": "P n n a", "Space Groups": ["Pnna"]},
        {"Code": 112, "Extinction Group": "P n n b", "Space Groups": ["Pnnb"]},
        {"Code": 113, "Extinction Group": "P n n n", "Space Groups": ["Pnnn"]},
        {
            "Code": 114,
            "Extinction Group": "C – – –",
            "Space Groups": ["C222", "Cmm2", "Cmmm", "Cm2m", "C2mm"],
        },
        {"Code": 115, "Extinction Group": "C – – 21", "Space Groups": ["C2221"]},
        {
            "Code": 116,
            "Extinction Group": "C – – (ab)",
            "Space Groups": ["Cm2a", "Cmma", "C2mb", "Cmmb"],
        },
        {
            "Code": 117,
            "Extinction Group": "C – c –",
            "Space Groups": ["Cmc21", "Cmcm", "C2cm"],
        },
        {
            "Code": 118,
            "Extinction Group": "C – c (ab)",
            "Space Groups": ["C2cb", "Cmca"],
        },
        {
            "Code": 119,
            "Extinction Group": "C c – –",
            "Space Groups": ["Ccm21", "Ccmm", "Cc2m"],
        },
        {
            "Code": 120,
            "Extinction Group": "C c – (ab)",
            "Space Groups": ["Cc2a", "Ccmb"],
        },
        {"Code": 121, "Extinction Group": "C c c –", "Space Groups": ["Ccc2", "Cccm"]},
        {
            "Code": 122,
            "Extinction Group": "C c c (ab)",
            "Space Groups": ["Ccca", "Cccb"],
        },
        {
            "Code": 123,
            "Extinction Group": "B – – –",
            "Space Groups": ["B222", "Bmm2", "Bmmm", "Bm2m", "B2mm"],
        },
        {"Code": 124, "Extinction Group": "B – 21 –", "Space Groups": ["B2212"]},
        {
            "Code": 125,
            "Extinction Group": "B – – b",
            "Space Groups": ["Bm21b", "Bmmb", "B2mb"],
        },
        {
            "Code": 126,
            "Extinction Group": "B – (ac)-",
            "Space Groups": ["Bma2", "Bmam", "B2cm", "Bmcm"],
        },
        {
            "Code": 127,
            "Extinction Group": "B – (ac)b",
            "Space Groups": ["B2cb", "Bmab"],
        },
        {
            "Code": 128,
            "Extinction Group": "B b – –",
            "Space Groups": ["Bbm2", "Bbmm", "Bb21m"],
        },
        {"Code": 129, "Extinction Group": "B b – b", "Space Groups": ["Bb2b", "Bbmb"]},
        {
            "Code": 130,
            "Extinction Group": "B b (ac)-",
            "Space Groups": ["Bba2", "Bbcm"],
        },
        {
            "Code": 131,
            "Extinction Group": "B b (ac)b",
            "Space Groups": ["Bbab", "Bbcb"],
        },
        {
            "Code": 132,
            "Extinction Group": "A – – –",
            "Space Groups": ["A222", "Amm2", "Ammm", "Am2m", "A2mm"],
        },
        {"Code": 133, "Extinction Group": "A 21 – –", "Space Groups": ["A2122"]},
        {
            "Code": 134,
            "Extinction Group": "A – – a",
            "Space Groups": ["Am2a", "Amma", "A21ma"],
        },
        {
            "Code": 135,
            "Extinction Group": "A – a –",
            "Space Groups": ["Ama2", "Amam", "A21am"],
        },
        {"Code": 136, "Extinction Group": "A – a a", "Space Groups": ["A2aa", "Amaa"]},
        {
            "Code": 137,
            "Extinction Group": "A(bc)- –",
            "Space Groups": ["Abm2", "Abmm", "Ac2m", "Acmm"],
        },
        {"Code": 138, "Extinction Group": "A(bc)- a", "Space Groups": ["Ac2a", "Abma"]},
        {"Code": 139, "Extinction Group": "A(bc)a –", "Space Groups": ["Aba2", "Acam"]},
        {"Code": 140, "Extinction Group": "A(bc)a a", "Space Groups": ["Abaa", "Acaa"]},
        {
            "Code": 141,
            "Extinction Group": "I – – –",
            "Space Groups": ["I222", "Imm2", "Immm", "I212121", "Im2m", "I2mm"],
        },
        {
            "Code": 142,
            "Extinction Group": "I – – (ab)",
            "Space Groups": ["Im2a", "Imma", "I2mb", "Immb"],
        },
        {
            "Code": 143,
            "Extinction Group": "I – (ac)-",
            "Space Groups": ["Ima2", "Imam", "I2cm", "Imcm"],
        },
        {"Code": 144, "Extinction Group": "I – c b", "Space Groups": ["I2cb", "Imcb"]},
        {
            "Code": 145,
            "Extinction Group": "I(bc)- –",
            "Space Groups": ["Ibm2", "Ibmm", "Ic2m", "Icmm"],
        },
        {"Code": 146, "Extinction Group": "I c – a", "Space Groups": ["Ic2a", "Icma"]},
        {"Code": 147, "Extinction Group": "I b a –", "Space Groups": ["Iba2", "Ibam"]},
        {
            "Code": 148,
            "Extinction Group": "I b c a",
            "Space Groups": ["Ibca", "Icab"],
        },  # I c a b is not a spacegroup. I added it to prevent a fail
        {
            "Code": 149,
            "Extinction Group": "F – – –",
            "Space Groups": ["F222", "Fmm2", "Fmmm", "Fm2m", "F2mm"],
        },
        {"Code": 150, "Extinction Group": "F – d d", "Space Groups": ["F2dd"]},
        {"Code": 151, "Extinction Group": "F d – d", "Space Groups": ["Fd2d"]},
        {"Code": 152, "Extinction Group": "F d d –", "Space Groups": ["Fdd2"]},
        {"Code": 153, "Extinction Group": "F d d d", "Space Groups": ["Fddd"]},
        {
            "Code": 154,
            "Extinction Group": "P – – –",
            "Space Groups": [
                "P4",
                "P-4",
                "P4/m",
                "P422",
                "P4mm",
                "P-42m",
                "P4/mmm",
                "P-4m2",
            ],
        },
        {
            "Code": 155,
            "Extinction Group": "P – 21 –",
            "Space Groups": ["P4212", "P-421m"],
        },
        {
            "Code": 156,
            "Extinction Group": "P 42 – –",
            "Space Groups": ["P42", "P42/m", "P4222"],
        },
        {"Code": 157, "Extinction Group": "P 42 21 –", "Space Groups": ["P42212"]},
        {
            "Code": 158,
            "Extinction Group": "P 41 – –",
            "Space Groups": ["P41", "P43", "P4122", "P4322"],
        },
        {
            "Code": 159,
            "Extinction Group": "P 41 21 –",
            "Space Groups": ["P41212", "P43212"],
        },
        {
            "Code": 160,
            "Extinction Group": "P – – c",
            "Space Groups": ["P42mc", "P-42c", "P42/mmc"],
        },
        {"Code": 161, "Extinction Group": "P – 21 c", "Space Groups": ["P-421c"]},
        {
            "Code": 162,
            "Extinction Group": "P – b –",
            "Space Groups": ["P4bm", "P-4b2", "P4/mbm"],
        },
        {
            "Code": 163,
            "Extinction Group": "P – b c",
            "Space Groups": ["P42bc", "P42/mbc"],
        },
        {
            "Code": 164,
            "Extinction Group": "P – c –",
            "Space Groups": ["P42cm", "P-4c2", "P42/mcm"],
        },
        {
            "Code": 165,
            "Extinction Group": "P – c c",
            "Space Groups": ["P4cc", "P4/mcc"],
        },
        {
            "Code": 166,
            "Extinction Group": "P – n –",
            "Space Groups": ["P42nm", "P-4n2", "P42/mnm"],
        },
        {
            "Code": 167,
            "Extinction Group": "P – n c",
            "Space Groups": ["P4nc", "P4/mnc"],
        },
        {
            "Code": 168,
            "Extinction Group": "P n – –",
            "Space Groups": ["P4/n", "P4/nmm"],
        },
        {"Code": 169, "Extinction Group": "P 42/n – –", "Space Groups": ["P42/n"]},
        {"Code": 170, "Extinction Group": "P n – c", "Space Groups": ["P42/nmc"]},
        {"Code": 171, "Extinction Group": "P n b –", "Space Groups": ["P4/nbm"]},
        {"Code": 172, "Extinction Group": "P n b c", "Space Groups": ["P42/nbc"]},
        {"Code": 173, "Extinction Group": "P n c –", "Space Groups": ["P42/ncm"]},
        {"Code": 174, "Extinction Group": "P n c c", "Space Groups": ["P4/ncc"]},
        {"Code": 175, "Extinction Group": "P n n –", "Space Groups": ["P42/nnm"]},
        {"Code": 176, "Extinction Group": "P n n c", "Space Groups": ["P4/nnc"]},
        {
            "Code": 177,
            "Extinction Group": "I – – –",
            "Space Groups": [
                "I4",
                "I-4",
                "I4/m",
                "I422",
                "I4mm",
                "I-42m",
                "I4/mmm",
                "I-4m2",
            ],
        },
        {"Code": 178, "Extinction Group": "I 41 – –", "Space Groups": ["I41", "I4122"]},
        {
            "Code": 179,
            "Extinction Group": "I – – d",
            "Space Groups": ["I41md", "I-42d"],
        },
        {
            "Code": 180,
            "Extinction Group": "I – c –",
            "Space Groups": ["I4cm", "I-4c2", "I4/mcm"],
        },
        {"Code": 181, "Extinction Group": "I – c d", "Space Groups": ["I41cd"]},
        {"Code": 182, "Extinction Group": "I 41/a – –", "Space Groups": ["I41/a"]},
        {"Code": 183, "Extinction Group": "I a – d", "Space Groups": ["I41/amd"]},
        {"Code": 184, "Extinction Group": "I a c d", "Space Groups": ["I41/acd"]},
        {
            "Code": 185,
            "Extinction Group": "P – – –",
            "Space Groups": [
                "P3",
                "P-3",
                "P321",
                "P3m1",
                "P-3m1",
                "P312",
                "P31m",
                "P-31m",
                "P6",
                "P-6",
                "P6/m",
                "P622",
                "P6mm",
                "P-62m",
                "P6/mmm",
                "P-6m2",
            ],
        },
        {
            "Code": 186,
            "Extinction Group": "P 31 – –",
            "Space Groups": ["P31", "P3121", "P3112", "P32", "P3221", "P3212"],
        },
        {
            "Code": 187,
            "Extinction Group": "P – – c",
            "Space Groups": ["P31c", "P-31c", "P63mc", "P-62c", "P63/mmc"],
        },
        {
            "Code": 188,
            "Extinction Group": "P – c –",
            "Space Groups": ["P3c1", "P-3c1", "P63cm", "P-6c2", "P63/mcm"],
        },
        {
            "Code": 189,
            "Extinction Group": "R (obv) – –",
            "Space Groups": ["R3", "R-3", "R32", "R3m", "R-3m"],
        },
        {
            "Code": 190,
            "Extinction Group": "R (obv)- – c",
            "Space Groups": ["R3c", "R-3c"],
        },
        {
            "Code": 191,
            "Extinction Group": "R (rev) – –",
            "Space Groups": ["R3", "R-3", "R32", "R3m", "R-3m"],
        },
        {
            "Code": 192,
            "Extinction Group": "R (rev)- – c",
            "Space Groups": ["R3c", "R-3c"],
        },
        {
            "Code": 193,
            "Extinction Group": "R – – –",
            "Space Groups": ["R3", "R-3", "R32", "R3m", "R-3m"],
        },
        {"Code": 194, "Extinction Group": "R – – c", "Space Groups": ["R3c", "R-3c"]},
        {
            "Code": 195,
            "Extinction Group": "P 63 – –",
            "Space Groups": ["P63", "P63/m", "P6322"],
        },
        {
            "Code": 196,
            "Extinction Group": "P 62 – –",
            "Space Groups": ["P62", "P6222", "P64", "P6422"],
        },
        {
            "Code": 197,
            "Extinction Group": "P 61 – –",
            "Space Groups": ["P61", "P6122", "P65", "P6522"],
        },
        {
            "Code": 198,
            "Extinction Group": "P – c c",
            "Space Groups": ["P6cc", "P6/mcc"],
        },
        {
            "Code": 199,
            "Extinction Group": "P – – –",
            "Space Groups": ["P23", "Pm-3", "P432", "P-43m", "Pm-3m"],
        },
        {"Code": 200, "Extinction Group": "P 21 – –", "Space Groups": ["P213"]},
        {"Code": 201, "Extinction Group": "P 42 – –", "Space Groups": ["P4232"]},
        {
            "Code": 202,
            "Extinction Group": "P 41 – –",
            "Space Groups": ["P4132", "P4332"],
        },
        {
            "Code": 203,
            "Extinction Group": "P – – n",
            "Space Groups": ["P-43n", "Pm-3n"],
        },
        {"Code": 204, "Extinction Group": "P a – –", "Space Groups": ["Pa-3"]},
        {"Code": 205, "Extinction Group": "P n – –", "Space Groups": ["Pn-3", "Pn-3m"]},
        {"Code": 206, "Extinction Group": "P n – n", "Space Groups": ["Pn-3n"]},
        {
            "Code": 207,
            "Extinction Group": "I – – –",
            "Space Groups": ["I23", "I213", "Im-3", "I432", "I-43m", "Im-3m"],
        },
        {"Code": 208, "Extinction Group": "I 41 – –", "Space Groups": ["I4132"]},
        {"Code": 209, "Extinction Group": "I – – d", "Space Groups": ["I-43d"]},
        {"Code": 210, "Extinction Group": "I a – –", "Space Groups": ["Ia-3"]},
        {"Code": 211, "Extinction Group": "I a – d", "Space Groups": ["Ia-3d"]},
        {
            "Code": 212,
            "Extinction Group": "F – – –",
            "Space Groups": ["F23", "Fm-3", "F432", "F-43m", "Fm-3m"],
        },
        {"Code": 213, "Extinction Group": "F 41 – –", "Space Groups": ["F4132"]},
        {
            "Code": 214,
            "Extinction Group": "F – – c",
            "Space Groups": ["F-43c", "Fm-3c"],
        },
        {"Code": 215, "Extinction Group": "F d – –", "Space Groups": ["Fd-3", "Fd-3m"]},
        {"Code": 216, "Extinction Group": "F d – c", "Space Groups": ["Fd-3c"]},
        {"Code": 217, "Extinction Group": "P –", "Space Groups": ["P1", "P-1"]},
    ]

    # Create a lookup dictionary for space groups
    spacegroup_symbol = spacegroup_symbol_hm.replace(" ", "")
    for row in table_data:
        if spacegroup_symbol in row["Space Groups"]:
            return row["Extinction Group"], row["Code"]
        elif spacegroup_symbol_hm in row["Space Groups"]:
            return row["Extinction Group"], row["Code"]
    else:
        print(f"{spacegroup_symbol_hm} {spacegroup_symbol} Not in lookup")
        return None, None
