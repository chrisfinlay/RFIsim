import numpy as np
import datetime
import pytz
import yaml
import os

def extract_timedelta(dt):
    """
    Convert from various types to a timedelta.

    Parameters:
    -----------
    dt: str, int, float
        The object to convert to a timedelta.

    Returns:
    --------
    timedelta: datetime.timedelta
        A timedelta object.
    """
    if type(dt) in [int, float]:
        return datetime.timedelta(seconds=dt)
    elif type(dt) is str:
        hh, mm, ss = dt.split(':')
        return datetime.timedelta(hours=int(hh), minutes=int(mm), seconds=float(ss))
    else:
        print('\nCould not convert {} to a datetime object.\n'.format(dt))


def load_file(input_dir, file_path):
    """
    Load a range of file types into a numpy array. e.g. .txt, .csv, .npy

    Parameters:
    -----------
    file: str
        Path to file that needs to be loaded.

    Returns:
    --------
    array: np.array
        Array loaded in from file.
    """
    file_type = os.path.splitext(file_path)[1]
    if input_dir:
        file_path = os.path.join(input_dir, file_path)

    if not file_type:
        return
    elif file_type=='.txt':
        array = np.loadtxt(file_path)
    elif file_type=='.csv':
        array = np.loadtxt(file_path, delimiter=',')
    elif file_type=='.npy':
        array = np.load(file_path)
    elif file_type=='.npz':
        fp = np.load(file_path)
        array = []
        print('\nConcatenating arrays on new axis 0.\n')
        for key in fp.keys():
            array.append(fp[key])
        array = np.array(array)
    else:
        print('\nCould not load {}\nFile type not understood. \nPlease use either .txt, .csv, .npy or .npz\n'.format(file_path))

    return array

def load_config(config_file='config.yml'):
    """
    Load the config file to populate parameter space.

    Parameters:
    -----------
    config_file: str
        Path to the YAML config file.

    Returns:
    --------
    config: dict
        Dictionary of configuration parameters.
    """
    config = yaml.safe_load(open(config_file))

    input_dir = config['process']['input_dir']

    # import and instantiate tzwhere to get timezone from GPS coordinate
#     from tzwhere import tzwhere
#     tzwhere = tzwhere.tzwhere()
    config['telescope']['time_zone'] = tzwhere.tzNameAt(*config['telescope']['GPS_coords'][:2])
    config['telescope']['GPS_coords'] = np.array(config['telescope']['GPS_coords'])
    config['telescope']['time_var'] = extract_timedelta(config['telescope']['time_var'])

#     load all telescope parameters into memory
    for x in ['ENU_coords', 'frequencies', 'bandpass_xx', 'bandpass_xy', 'bandpass_yx', 'bandpass_yy']:
        config['telescope'][x] = load_file(input_dir, config['telescope'][x])


    config['observation']['target'] = np.array(config['observation']['target'])
#     Set timezone aware datetime for observation start
    if config['observation']['start_time']:
        if config['observation']['start_time'] == 'transit':
            config['observation']['start_datetime'] = None # Yet to be implemented
        else:
            config['observation']['start_time'] = extract_timedelta(config['observation']['start_time'])
            config['observation']['start_datetime'] = datetime.datetime.strptime(config['observation']['start_date'], '%d/%m/%Y')
            config['observation']['start_datetime'] = pytz.timezone(config['telescope']['time_zone']).localize(config['observation']['start_datetime'])
            config['observation']['start_datetime'] += config['observation']['start_time']
    else:
        print('\nNo start time set. If you are unsure simply use "transit".\n')

    config['observation']['int_time'] = extract_timedelta(config['observation']['int_time'])

    if config['observation']['time_steps']:
        config['observation']['duration'] = config['observation']['time_steps'] * \
                                            config['observation']['int_time']
    else:
        config['observation']['duration'] = extract_timedelta(config['observation']['duration'])

#     Load astronomical sky model

#     Load RFI parameters
    config['rfi']['freq_dist'] = load_file(input_dir, config['rfi']['freq_dist'])
    for x in ['satellites', 'cell_towers', 'planes']:
        try:
            config['rfi'][x]['time_var'] = extract_timedelta(config['rfi'][x]['time_var'])
        except:
            print('\nNo time variability set for {}.'.format(x))
        try:
            config['rfi'][x]['freqs'] = load_file(input_dir, config['rfi'][x]['freqs'])
        except:
            print('\nNo frequency ranges set for {}.'.format(x))

    try:
        config['rfi']['cell_towers']['GPS_coords'] = load_file(input_dir, config['rfi']['cell_towers']['GPS_coords'])
    except:
        print('\nNo GPS coordinates set for cell towers.')

    try:
        config['rfi']['planes']['paths'] = load_file(input_dir, config['rfi']['planes']['paths'])
    except:
        print('\nNo flight paths set for planes.')

    return config
