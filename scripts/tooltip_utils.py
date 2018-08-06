from __future__ import division
from rdkit.Chem import Draw
from rdkit import Chem
import io
import base64
import numpy as np
from scipy.misc import toimage, imsave, imresize

def smi2image(smiles):
    """Turn a smiles string into an image"""
    molecule = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(molecule, size=[150,150])

def molecule_tooltips(smiles_list):
    """Creates an array of molecule drawings that can be used as tooltips in KeplerMapper"""
    tooltips = []
    n_molecules = len(smiles_list)
    for smiles_count, smiles_l in enumerate(smiles_list):
        if type(smiles_l) == np.ndarray:
            smiles = smiles_l[0]
        else:
            smiles = smiles_l
        if smiles_count % 1000 == 0:
            print(smiles_count)
        output = io.BytesIO()
        img = toimage(smi2image(smiles))
        img.save(output, format='PNG')
        contents = output.getvalue()
        img_encoded = base64.b64encode(contents)
        img_tag = """<p>
                     <div style="width:150px;
                                height:150px;
                                overflow:hidden;
                                float:left;
                                position:relative;">
                     <img src="data:image/png;base64,%s" style="position:absolute; top:0; right:0;
                                                                width: 150px; height: 150px;" />
                     </div>
                     </p>""" % (img_encoded.decode('utf-8'))
        tooltips.append(img_tag)
        output.close()
    tooltips = np.array(tooltips)
    return tooltips

# def molecule_tooltips_2(smiles_list):
#     tooltips=[]
#     n_molecules = len(smiles_list)
#     for smiles_count, smiles in enumerate(smiles_list):
#         img = toimage(imresize(smi2image(smiles), (64, 54)))
#         tooltips.append(img)
#         tooltips = np.array(tooltips)
#     return tooltips