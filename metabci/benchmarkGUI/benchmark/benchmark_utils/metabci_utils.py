from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.signal import tf2zpk  
    from scipy.signal import sosfiltfilt
    from metabci.brainda.paradigms import SSVEP, P300, MotorImagery, aVEP
    from metabci.brainda.algorithms.decomposition import generate_filterbank


def channel_selection(key):
    if key == 'occipital_9':
        return ['PZ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
    if key == 'occipital_8':
        return ["PO7", "PO3", "POZ", "PO4", "PO8", "O1", "OZ", "O2"]


def create_ssvep_paradigm(srate, channels, interval, events):
    """This paradigm will automatically apply 8-90Hz bandpass filtering to all input data"""
    paradigm = SSVEP(
        srate=srate,
        channels=channels,
        intervals=interval,
        events=events)

    def ssvep_datahook(X, y, meta, caches):
        filters = generate_filterbank(
            [[8, 90]], [[6, 95]], srate, order=4, rp=1 
        )
        X = sosfiltfilt(filters[0], X, axis=-1)
        return X, y, meta, caches


    paradigm.register_data_hook(ssvep_datahook)

    return paradigm


def gen_ssvep_filterbank(srate, n_bands=3):
    wp = [[8*i, 90] for i in range(1, n_bands+1)]
    ws = [[8*i-2, 95] for i in range(1, n_bands+1)]
    filterbank = generate_filterbank(
        wp, ws, srate, order=4, rp=1)
    filterweights = np.arange(1, len(filterbank)+1)**(-1.25) + 0.25
    return filterbank, filterweights


def create_mi_paradigm(srate, channels, interval, events):
    """该 paradigm 会自动对所有输入数据进行 8-90Hz 的带通滤波"""
    paradigm = MotorImagery(
        srate=srate,
        channels=channels,
        intervals=interval,
        events=events)

    def mi_datahook(X, y, meta, caches):
        filters = generate_filterbank(
            [[8, 90]], [[6, 95]], srate, order=4, rp=1
        )
        X = sosfiltfilt(filters[0], X, axis=-1)
    
    # add 6-30Hz bandpass filter in raw hook
    def mi_raw_hook(raw, caches):
        # do something with raw object
        raw.filter(6, 30, l_trans_bandwidth=2,h_trans_bandwidth=5,
            phase='zero-double')
        caches['raw_stage'] = caches.get('raw_stage', -1) + 1
        return raw, caches

    def mi_epochs_hook(epochs, caches):
        # do something with epochs object
        print(epochs.event_id)
        caches['epoch_stage'] = caches.get('epoch_stage', -1) + 1
        return epochs, caches

    def mi_data_hook(X, y, meta, caches):
        # retrive caches from the last stage
        print("Raw stage:{},Epochs stage:{}".format(caches['raw_stage'], caches['epoch_stage']))
        # do something with X, y, and meta
        caches['data_stage'] = caches.get('data_stage', -1) + 1
        return X, y, meta, caches
    
    
    paradigm.register_raw_hook(mi_raw_hook)
    paradigm.register_epochs_hook(mi_epochs_hook)
    paradigm.register_data_hook(mi_data_hook)


    return paradigm


def gen_mi_filterbank(srate, n_bands=3):
    # wp = [[4*i, 4*(i+1)] for i in range(1, n_bands+1)]
    # ws = [[8*i-2, 95] for i in range(1, n_bands+1)]
    wp=[(4,8),(8,12),(12,30)]
    ws=[(2,10),(6,14),(10,32)]
    filterbank = generate_filterbank(
        wp, ws, srate, order=4, rp=0.5)
    filterweights = np.arange(1, len(filterbank)+1)**(-1.25) + 0.25
    return filterbank, filterweights


def p300_channel_selection(key):
    if key == 'occipital_8':
        return ["PO7", "PO3", "PZ", "PO4", "PO8", "O1", "OZ", "O2"]
    
def create_p300_paradigm(channels):
    """This paradigm will automatically perform bandpass filtering of 0.1-100Hz and notch filtering of 50Hz on all input data, 
    in accordance with the original acquisition settings of Cattan-P300"""
    paradigm = P300(channels=channels)

    def p300_datahook(X, y, meta, caches):

        filters = generate_filterbank(
            [[0.1, 100]], [[0.05, 110]], 500, order=4, rp=1
        )
        X = sosfiltfilt(filters[0], X, axis=-1)

        from scipy.signal import iirnotch, zpk2sos
        w0 = 50/(500/2)  
        b, a = iirnotch(w0, Q=35)  
        sos = zpk2sos(*tf2zpk(b, a))  
        X = sosfiltfilt(sos, X, axis=-1)  
        return X, y, meta, caches
    
    def p300_raw_hook(raw, caches):
    # do something with raw continuous data
        caches['raw_stage'] = caches.get('raw_stage', -1) + 1
        return raw, caches

    def p300_trial_hook(raw, caches):
        # do something with trail raw object.
        raw.filter(1, 20, l_trans_bandwidth=1, h_trans_bandwidth=4,
                phase='zero-double')
        caches['trial_stage'] = caches.get('trial_stage', -1) + 1
        return raw, caches

    def p300_epochs_hook(epochs, caches):
        # do something with epochs object
        caches['epoch_stage'] = caches.get('epoch_stage', -1) + 1
        return epochs, caches

    def p300_data_hook(X, y, meta, caches):
        caches['data_stage'] = caches.get('data_stage', -1) + 1
        return X, y, meta, caches

    paradigm.register_raw_hook(p300_raw_hook)
    paradigm.register_trial_hook(p300_trial_hook)
    paradigm.register_epochs_hook(p300_epochs_hook)
    paradigm.register_data_hook(p300_data_hook)

    return paradigm


def gen_p300_filterbank(srate, n_bands=3):

    wp = [[0.1, 8], [0.1, 15], [0.1, 30]][:n_bands]
    ws = [[0.05, 10], [0.05, 18], [0.05, 35]][:n_bands]
    filterbank = generate_filterbank(
        wp, ws, srate, order=4, rp=1
    )
    filterweights = np.ones(len(filterbank))  
    return filterbank, filterweights

def create_avep_paradigm(srate, channels):

    paradigm = aVEP(
        srate=srate,
        channels=channels,)
    
    def avep_raw_hook(raw, caches):
        # do something with raw continuous data
        caches['raw_stage'] = caches.get('raw_stage', -1) + 1
        return raw, caches

    def avep_trial_hook(raw, caches):
        # do something with trail raw object.
        raw.filter(2, 21, l_trans_bandwidth=1, h_trans_bandwidth=4,
                phase='zero-double')
        caches['trial_stage'] = caches.get('trial_stage', -1) + 1
        return raw, caches

    def avep_epochs_hook(epochs, caches):
        # do something with epochs object
        caches['epoch_stage'] = caches.get('epoch_stage', -1) + 1
        return epochs, caches

    def avep_data_hook(X, y, meta, caches):
        caches['data_stage'] = caches.get('data_stage', -1) + 1
        return X, y, meta, caches

    paradigm.register_raw_hook(avep_raw_hook)
    paradigm.register_trial_hook(avep_trial_hook)
    paradigm.register_epochs_hook(avep_epochs_hook)
    paradigm.register_data_hook(avep_data_hook)

    return paradigm
