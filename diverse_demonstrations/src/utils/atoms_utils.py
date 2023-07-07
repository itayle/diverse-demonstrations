from typing import List


ATOMS_TO_IGNORE = {',', '(', ')', '<s>', ':', '#', '"'}


def get_atoms(target: str) -> List:
    atoms = tokenize_lf(target)
    atoms = [atom for atom in atoms if atom not in ATOMS_TO_IGNORE]

    return atoms

def tokenize_lf(lf, add_sos=True):
    if not lf:
        return []
    target = lf.replace('[ ', '[').replace(' ]', ']').replace("(", " ( ").replace(")", " ) ").replace(",", " , ")
    if add_sos:
        target = f"<s> ( {target} )"
    tokens = target.split()

    return tokens
