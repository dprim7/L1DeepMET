import awkward as ak # type: ignore

def to_np_array(ak_array, maxN=100, pad=0):
    return ak.fill_none(ak.pad_none(ak_array, maxN, clip=True, axis=-1), pad).to_numpy()