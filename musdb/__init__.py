from .audio_classes import MultiTrack, Source, Target
from os import path as op
import stempeg
import urllib.request
import collections
import numpy as np
import functools
import zipfile
import yaml
#import musdb #How can this make sense? Why am I importing musdb
import errno
import os


class DB(object):
    """
    The musdb DB Object

    Parameters
    ----------
    root : str, optional
        musdb Root path. If set to `None` it will be read
        from the `MUSDB_PATH` environment variable

    subsets : str or list, optional
        select a _musdb_ subset `train` or `test` (defaults to both)

    is_wav : boolean, optional
        expect subfolder with wav files for each source instead stems,
        defaults to `False`

    download : boolean, optional
        download sample version of MUSDB18 which includes 7s excerpts,
        defaults to `False`

    subsets : list[str], optional
        select a _musdb_ subset `train` or `test`.
        Default `None` loads `['train', 'test']`

    split : str, optional
        when `subsets=train`, `split` selects the train/validation split.
        `split='train' loads the training split, `split='valid'` loads the validation
        split. `split=None` applies no splitting.

    Attributes
    ----------
    setup_file : str
        path to yaml file. default: `setup.yaml`
    root : str
        musdb Root path. Default is `MUSDB_PATH`. In combination with
        `download`, this path will set the download destination and set to
        '~/musdb/' by default.
    sources_dir : str
        path to Sources directory
    sources_names : list[str]
        list of names of available sources
    targets_names : list[str]
        list of names of available targets
    setup : Dict
        loaded yaml configuration
    sample_rate : Optional(Float)
        sets sample rate for optional resampling. Defaults to none
        which results in `44100.0`

    Methods
    -------
    load_mus_tracks()
        Iterates through the musdb folder structure and
        returns ``Track`` objects

    """
    def __init__(
        self,
        root=None, # local path to the leakage removal dataset
        setup_file=None,
        is_wav=False,
        subsets=['train', 'test'],
        split=None,
        sample_rate=None, 
        data_path=None, 
        instrument='drums', 
    ):
        if root is None:
            raise RuntimeError("Variable `MUSDB_PATH` has not been set.")
        else:
            self.root = os.path.expanduser(root)

        if setup_file is not None:
            setup_path = op.join(self.root, setup_file)
        else:
            setup_path = os.path.join(
                root, 'configs', 'mus.yaml'
            )

        with open(setup_path, 'r') as f:
            self.setup = yaml.safe_load(f)

        if sample_rate != self.setup['sample_rate']:
            self.sample_rate = sample_rate

        self.sources_names = list(self.setup['sources'].keys())
        self.targets_names = list(self.setup['targets'].keys())
        self.is_wav = is_wav
        self.data_path = data_path
        self.instrument = instrument
        self.tracks = self.load_mus_tracks(subsets=subsets, split=split)

    def __getitem__(self, index):
        return self.tracks[index]

    def __len__(self):
        return len(self.tracks)

    #If needed, I can re-copy the get-validation-track-indeces function
    def get_track_indices_by_names(self, names):
        """Returns musdb track indices by track name

        Can be used to filter the musdb tracks for 
        a validation subset by trackname

        Parameters
        == == == == ==
        names : list[str], optional
            select tracks by a given `str` or list of tracknames

        Returns
        -------
        list[int]
            return a list of ``Track`` Objects
        """
        if isinstance(names, str):
            names = [names]
        
        return [[t.name for t in self.tracks].index(name) for name in names]

    #I will not delete the stuff related to train, but this code is not meant to be used for anything other than test.
    def load_mus_tracks(self, subsets=None, split=None):
        """Parses the musdb folder structure, returns list of `Track` objects

        Parameters
        ==========
        subsets : list[str], optional
            select a _musdb_ subset `train` or `test`.
            Default `None` loads [`train, test`].
        split : str
            for subsets='train', `split='train` applies a train/validation split.
            if `split='valid`' the validation split of the training subset will be used


        Returns
        -------
        list[Track]
            return a list of ``Track`` Objects
        """

        if subsets is not None:
            if isinstance(subsets, str):
                subsets = [subsets]
        else:
            subsets = ['train', 'test']

        if subsets != ['train'] and split is not None:
            raise RuntimeError("Subset has to set to `train` when split is used")

        tracks = []
        for subset in subsets:
            subset_folder = op.join(self.data_path, subset)
            for _, folders, files in os.walk(op.join(subset_folder, self.instrument)):
                if self.is_wav:
                    # parse pcm tracks and sort by name
                    for track_name in sorted(folders):
                        if subset == 'train':
                            if split == 'train' and track_name in self.setup['validation_tracks']:
                                continue
                            elif split == 'valid' and track_name not in self.setup['validation_tracks']:
                                continue

                        track_folder = op.join(subset_folder, self.instrument, track_name)
                        # track_number (in the modified leakage removal, there are additional subfolders from 0 - 9
                        for lvl2_, lvl2_folders, lvl2_files in os.walk(track_folder): 
                            for folder_num in sorted(lvl2_folders): 
                                # create new mus track
                                track = MultiTrack(
                                    name=track_name,
                                    path=op.join(
                                        track_folder,
                                        folder_num,
                                        self.setup['mixture']
                                    ),
                                    subset=subset,
                                    is_wav=self.is_wav,
                                    stem_id=self.setup['stem_ids']['mixture'],
                                    sample_rate=self.sample_rate
                                )

                                # add sources to track
                                sources = {}
                                for src, source_file in list(
                                    self.setup['sources'].items()
                                ):
                                    # create source object
                                    abs_path = op.join(
                                        track_folder,
                                        folder_num, 
                                        source_file
                                    )
                                    if os.path.exists(abs_path):
                                        sources[src] = Source(
                                        track,
                                        name=src,
                                        path=abs_path,
                                        stem_id=self.setup['stem_ids'][src],
                                        sample_rate=self.sample_rate
                                    )
                                track.sources = sources
                                track.targets = self.create_targets(track)
                                # add track to list of tracks
                                tracks.append(track)

        return tracks

    def create_targets(self, track):
        # add targets to track
        targets = collections.OrderedDict()
        for name, target_srcs in list(
            self.setup['targets'].items()
        ):
            # add a list of target sources
            target_sources = []
            for source, gain in list(target_srcs.items()):
                if source in list(track.sources.keys()):
                    # add gain to source tracks
                    track.sources[source].gain = float(gain)
                    # add tracks to components
                    target_sources.append(track.sources[source])
                    # add sources to target
            if target_sources:
                targets[name] = Target(
                    track,
                    sources=target_sources,
                    name=name
                )
        return targets

    def save_estimates(
        self,
        user_estimates,
        track,
        estimates_dir,
        write_stems=False
    ):
        """Writes `user_estimates` to disk while recreating the musdb file structure in that folder.

        Parameters
        ==========
        user_estimates : Dict[np.array]
            the target estimates.
        track : Track,
            musdb track object
        estimates_dir : str,
            output folder name where to save the estimates.
        """
        track_estimate_dir = op.join(
            estimates_dir, track.subset, track.name
        )
        if not os.path.exists(track_estimate_dir):
            os.makedirs(track_estimate_dir)

        # write out tracks to disk
        if write_stems:
            pass
            # to be implemented
        else:
            for target, estimate in list(user_estimates.items()):
                target_path = op.join(track_estimate_dir, target + '.wav')
                stempeg.write_audio(
                    path=target_path,
                    data=estimate,
                    sample_rate=track.rate
                )

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, "train"))

    #Maybe I should add a sample downloader for the leakage removal dataset
