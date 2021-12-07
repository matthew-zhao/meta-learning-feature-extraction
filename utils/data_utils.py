import collections
import cv2
import enum
import functools
import io
import logging
import os, glob
import json
import pickle
import six
import struct
import random
from PIL import Image
from PIL import ImageEnhance
from sklearn import preprocessing
import imageio
import numpy as np
import google_drive_downloader as gdd
import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, Dataset, IterableDataset, get_worker_info, sampler

from pathlib import Path
from six.moves import cPickle as pkl
from typing import Union, List, Set, Any, Dict

from . import example_pb2

logger = logging.getLogger(__name__)

_OMNIGLOT_BASE_PATH = './omniglot_resized'
_OMNIGLOT_GDD_FILE_ID = '1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI'
_MINIIMAGENET_BASE_PATH = './data/miniImageNet'
_TIEREDIMAGENET_BASE_PATH = './data/tieredImageNet'

OMNIGLOT_NUM_TRAIN_CLASSES = 1100
OMNIGLOT_NUM_VAL_CLASSES = 100
OMNIGLOT_NUM_TEST_CLASSES = 423
OMNIGLOT_NUM_SAMPLES_PER_CLASS = 20

class Split(enum.Enum):
    """The possible data splits."""
    TRAIN = 0
    VALID = 1
    TEST = 2

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)


class ImageJitter(object):
    def __init__(self, transformdict):
        transformtypedict = dict(Brightness=ImageEnhance.Brightness,
                                 Contrast=ImageEnhance.Contrast,
                                 Sharpness=ImageEnhance.Sharpness,
                                 Color=ImageEnhance.Color)
        self.params = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.params))

        for i, (transformer, alpha) in enumerate(self.params):
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out

def cycle_(iterable):
    # Creating custom cycle since itertools.cycle attempts to save all outputs in order to
    # re-cycle through them, creating amazing memory leak
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

def parse_record(feat_dic):
    # get BGR image from bytes
    image = cv2.imdecode(feat_dic["image"], -1)
    # from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    feat_dic["image"] = image
    return feat_dic


def get_transforms(data_config: 'DataConfig',
                   split: Split):
    if split == Split["TRAIN"]:
        return train_transform(data_config)
    else:
        return test_transform(data_config)


def test_transform(data_config: 'DataConfig'):
    # resize_size = int(data_config.image_size * 256 / 224)
    # assert resize_size == data_config.image_size * 256 // 224
    resize_size = data_config.image_size

    transf_dict = {'resize': transforms.Resize((resize_size, resize_size)),
                   'center_crop': transforms.CenterCrop(data_config.image_size),
                   'to_tensor': transforms.ToTensor(),
                   'normalize': normalize}
    augmentations = data_config.test_transforms

    return transforms.Compose([transf_dict[key] for key in augmentations])


def train_transform(data_config: 'DataConfig'):
    transf_dict = {'resize': transforms.Resize((data_config.image_size, data_config.image_size)),
                   'center_crop': transforms.CenterCrop(data_config.image_size),
                   'random_resized_crop': transforms.RandomResizedCrop(data_config.image_size),
                   'jitter': ImageJitter(jitter_param),
                   'random_flip': transforms.RandomHorizontalFlip(),
                   'to_tensor': transforms.ToTensor(),
                   'normalize': normalize}
    augmentations = data_config.train_transforms

    return transforms.Compose([transf_dict[key] for key in augmentations])


def tfrecord_iterator(data_path: str,
                      random_gen,
                      index_path = None,
                      shard = None,
                      shuffle: bool = False,
                      ):
    """Create an iterator over the tfrecord dataset.
    Since the tfrecords file stores each example as bytes, we can
    define an iterator over `datum_bytes_view`, which is a memoryview
    object referencing the bytes.
    Params:
    -------
    data_path: str
        TFRecord file path.
    index_path: str, optional, default=None
        Index file path. Can be set to None if no file is available.
    shard: tuple of ints, optional, default=None
        A tuple (index, count) representing worker_id and num_workers
        count. Necessary to evenly split/shard the dataset among many
        workers (i.e. >1).
    Yields:
    -------
    datum_bytes_view: memoryview
        Object referencing the specified `datum_bytes` contained in the
        file (for a single record).
    """
    file = io.open(data_path, "rb")

    length_bytes = bytearray(8)
    crc_bytes = bytearray(4)
    datum_bytes = bytearray(1024 * 1024)

    def random_reader(indexes: np.ndarray,
                      random_gen: np.random.RandomState):
        random_permutation = random_gen.permutation(range(indexes.shape[0]))
        for i in random_permutation:
            start = indexes[i, 0]
            end = indexes[i, 0] + indexes[i, 1]
            yield from read_records(start, end)

    def read_records(start_offset=None, end_offset=None):
        nonlocal length_bytes, crc_bytes, datum_bytes

        if start_offset is not None:
            file.seek(start_offset)
        if end_offset is None:
            end_offset = os.path.getsize(data_path)
        while file.tell() < end_offset:
            if file.readinto(length_bytes) != 8:
                raise RuntimeError("Failed to read the record size.")
            if file.readinto(crc_bytes) != 4:
                raise RuntimeError("Failed to read the start token.")
            length, = struct.unpack("<Q", length_bytes)
            if length > len(datum_bytes):
                datum_bytes = datum_bytes.zfill(int(length * 1.5))
            datum_bytes_view = memoryview(datum_bytes)[:length]
            if file.readinto(datum_bytes_view) != length:
                raise RuntimeError("Failed to read the record.")
            if file.readinto(crc_bytes) != 4:
                raise RuntimeError("Failed to read the end token.")
            yield datum_bytes_view, (start_offset, end_offset)

    if index_path is None:
        raise ValueError("Index files need to be provided")
    else:
        indexes = np.loadtxt(index_path, dtype=np.int64)
        # if shard is None:
        if shuffle:
            yield from random_reader(indexes=indexes, random_gen=random_gen)
        else:
            yield from read_records()

    file.close()

def process_feature(feature: example_pb2.Feature,
                    typename: str,
                    typename_mapping: dict,
                    key: str):
    # NOTE: We assume that each key in the example has only one field
    # (either "bytes_list", "float_list", or "int64_list")!
    field = feature.ListFields()[0]
    inferred_typename, value = field[0].name, field[1].value

    if typename is not None:
        tf_typename = typename_mapping[typename]
        if tf_typename != inferred_typename:
            reversed_mapping = {v: k for k, v in typename_mapping.items()}
            raise TypeError(f"Incompatible type '{typename}' for `{key}` "
                        f"(should be '{reversed_mapping[inferred_typename]}').")

    if inferred_typename == "bytes_list":
        value = np.frombuffer(value[0], dtype=np.uint8)
    elif inferred_typename == "float_list":
        value = np.array(value, dtype=np.float32)
    elif inferred_typename == "int64_list":
        value = np.array(value, dtype=np.int32)
    return value


def extract_feature_dict(features, description, typename_mapping):
    if isinstance(features, example_pb2.FeatureLists):
        features = features.feature_list

        def get_value(typename, typename_mapping, key):
            feature = features[key].feature
            fn = functools.partial(process_feature, typename=typename,
                                   typename_mapping=typename_mapping, key=key)
            return list(map(fn, feature))
    elif isinstance(features, example_pb2.Features):
        features = features.feature

        def get_value(typename, typename_mapping, key):
            return process_feature(features[key], typename,
                                   typename_mapping, key)
    else:
        raise TypeError(f"Incompatible type: features should be either of type "
                        f"example_pb2.Features or example_pb2.FeatureLists and "
                        f"not {type(features)}")

    all_keys = list(features.keys())

    if description is None or len(description) == 0:
        description = dict.fromkeys(all_keys, None)
    elif isinstance(description, list):
        description = dict.fromkeys(description, None)

    processed_features = {}
    for key, typename in description.items():
        if key not in all_keys:
            raise KeyError(f"Key {key} doesn't exist (select from {all_keys})!")

        processed_features[key] = get_value(typename, typename_mapping, key)

    return processed_features

def example_loader(data_path: str,
                   random_gen,
                   index_path,
                   description = None,
                   shard = None,
                   shuffle: bool = False,
                   ):
    """Create an iterator over the (decoded) examples contained within
    the dataset.
    Decodes raw bytes of the features (contained within the dataset)
    into its respective format.
    Params:
    -------
    data_path: str
        TFRecord file path.
    index_path: str or None
        Index file path. Can be set to None if no file is available.
    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.
    shard: tuple of ints, optional, default=None
        A tuple (index, count) representing worker_id and num_workers
        count. Necessary to evenly split/shard the dataset among many
        workers (i.e. >1).
    Yields:
    -------
    features: dict of {str, np.ndarray}
        Decoded bytes of the features into its respective data type (for
        an individual record).
    """

    typename_mapping = {
        "byte": "bytes_list",
        "float": "float_list",
        "int": "int64_list"
    }

    record_iterator = tfrecord_iterator(data_path, random_gen, index_path, shard, shuffle)

    for record, (start, end) in record_iterator:
        # yield record
        example = example_pb2.Example()
        example.ParseFromString(record)
        feature_dic = extract_feature_dict(example.features, description, typename_mapping)
        feature_dic['id'] = start
        yield feature_dic

def tfrecord_loader(data_path: str,
                    index_path,
                    random_gen,
                    description = None,
                    shard = None,
                    shuffle = False,
                    sequence_description = None,
                    ):
    """Create an iterator over the (decoded) examples contained within
    the dataset.
    Decodes raw bytes of the features (contained within the dataset)
    into its respective format.
    Params:
    -------
    data_path: str
        TFRecord file path.
    index_path: str or None
        Index file path. Can be set to None if no file is available.
    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        or an empty list or dictionary, then all features contained in
        the file are extracted.
    shard: tuple of ints, optional, default=None
        A tuple (index, count) representing worker_id and num_workers
        count. Necessary to evenly split/shard the dataset among many
        workers (i.e. >1).
    sequence_description: list or dict of str, optional, default=None
        Similar to `description`, but refers to the sequence features
        within a `SequenceExample`. When this field is `None`, then it
        is assumed that an `Example` is being read otherwise, a
        `SequenceExample` is read. If an empty list or dictionary is
        passed, then all features contained in the file are extracted.
    Yields:
    -------
    features: dict of {str, value}
        Decoded bytes of the features into its respective data type (for
        an individual record). `value` is either going to be an np.ndarray
        in the instance of an `Example` and a list of np.ndarray in the
        instance of a `SequenceExample`.
    """
    return example_loader(data_path, random_gen, index_path, description, shard, shuffle)

def get_classes(split: Split, classes_per_split: Dict[Split, int]):
    """Gets the sequence of class labels for a split.
    Class id's are returned ordered and without gaps.
    Args:
        split: A Split, the split for which to get classes.
        classes_per_split: Matches each Split to the number of its classes.
    Returns:
        The sequence of classes for the split.
    Raises:
        ValueError: An invalid split was specified.
    """
    num_classes = classes_per_split[split]

    # Find the starting index of classes for the given split.
    if split == Split.TRAIN:
        offset = 0
    elif split == Split.VALID:
        offset = classes_per_split[Split.TRAIN]
    elif split == Split.TEST:
        offset = (classes_per_split[Split.TRAIN] + classes_per_split[Split.VALID])
    else:
        raise ValueError('Invalid dataset split.')

    # Get a contiguous range of classes from split.
    return range(offset, offset + num_classes)


def _check_validity_of_restricted_classes_per_split(
        restricted_classes_per_split: Dict[Split, int],
        classes_per_split: Dict[Split, int]):
    """Check the validity of the given restricted_classes_per_split.
    Args:
        restricted_classes_per_split: A dict mapping Split enums to the number of
            classes to restrict to for that split.
        classes_per_split: A dict mapping Split enums to the total available number
            of classes for that split.
    Raises:
        ValueError: if restricted_classes_per_split is invalid.
    """
    for split_enum, num_classes in restricted_classes_per_split.items():
        if split_enum not in [Split.TRAIN,
                              Split.VALID,
                              Split.TEST]:
            raise ValueError('Invalid key {} in restricted_classes_per_split.'
                             'Valid keys are: Split.TRAIN, '
                             'Split.VALID, and '
                             'Split.TEST'.format(split_enum))
        if num_classes > classes_per_split[split_enum]:
            raise ValueError('restricted_classes_per_split can not specify a '
                             'number of classes greater than the total available '
                             'for that split. Specified {} for split {} but have '
                             'only {} available for that split.'.format(
                                 num_classes,
                                 split_enum,
                                 classes_per_split[split_enum]))

def get_total_images_per_class(data_spec, class_id: int = None):
    """Returns the total number of images of a class in a data_spec and pool.
    Args:
        data_spec: A DatasetSpecification, or BiLevelDatasetSpecification.
        class_id: The class whose number of images will be returned. If this is
            None, it is assumed that the dataset has the same number of images for
            each class.
        pool: A string ('train' or 'test', optional) indicating which example-level
            split to select, if the current dataset has them.
    Raises:
        ValueError: when
            - no class_id specified and yet there is class imbalance, or
            - no pool specified when there are example-level splits, or
            - pool is specified but there are no example-level splits, or
            - incorrect value for pool.
        RuntimeError: the DatasetSpecification is out of date (missing info).
    """
    if class_id is None:
        if len(set(data_spec.images_per_class.values())) != 1:
            raise ValueError('Not specifying class_id is okay only when all classes'
                             ' have the same number of images')
        class_id = 0

    if class_id not in data_spec.images_per_class:
        raise RuntimeError('The DatasetSpecification should be regenerated, as '
                           'it does not have a non-default value for class_id {} '
                           'in images_per_class.'.format(class_id))
    num_images = data_spec.images_per_class[class_id]

    return num_images

def load_image(file_path):
    """Loads and transforms an Omniglot image.

    Args:
        file_path (str): file path of image

    Returns:
        a Tensor containing image data
            shape (1, 28, 28)
    """
    x = imageio.imread(file_path)
    x = torch.tensor(x, dtype=torch.float32).reshape([1, 28, 28])
    x = x / 255.0
    return 1 - x

class OmniglotDataset(Dataset):
    """Omniglot dataset for meta-learning.

    Each element of the dataset is a task. A task is specified with a key,
    which is a tuple of class indices (no particular order). The corresponding
    value is the instantiated task, which consists of sampled (image, label)
    pairs.
    """

    _BASE_PATH = './omniglot_resized'
    _GDD_FILE_ID = '1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI'

    def __init__(self, character_folders, partition):
        """Inits OmniglotDataset.

        Args:
            num_support (int): number of support examples per class
            num_query (int): number of query examples per class
        """
        super().__init__()

        # # get all character folders
        self._character_images = glob.glob(
            os.path.join(_OMNIGLOT_BASE_PATH, '*/*/*'))

        le = preprocessing.LabelEncoder()
        le.fit_transform(character_folders)

        # check problem arguments
        self._character_folders = character_folders
        self._label_mapping = le

    def __getitem__(self, class_img_pair_index):
        """Gets an image and its corresponding label

        Data for each class is sampled uniformly at random without replacement.
        The ordering of the labels corresponds to that of class_idxs.

        Args:
            class_idxs (tuple[int]): class indices that comprise the task

        Returns:
            image, label
        """
        class_idx, img_index = class_img_pair_index
        all_file_paths = glob.glob(
            os.path.join(self._character_folders[class_idx], '*.png')
        )
        image = load_image(all_file_paths[img_index])
        label = self._character_folders[class_idx]
        label_as_tensor = torch.as_tensor(self._label_mapping.transform([label]))
        return image, label_as_tensor, "0-%s-%s.jpeg" % (label_as_tensor.item(), all_file_paths[img_index])

    def __len__(self):
        return len(self._character_images)

class MiniImageNet(Dataset):
    def __init__(self, data_root, data_aug=False, partition='train', pretrain=True, transform=None):
        super(Dataset, self).__init__()
        self.data_root = data_root
        self.partition = partition
        self.data_aug = data_aug
        self.mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        self.std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.pretrain = pretrain

        if transform is None:
            if self.partition == 'train' and self.data_aug:
                self.transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.RandomCrop(84, padding=8),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
            else:
                self.transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.transform = transform

        if self.pretrain:
            self.file_pattern = 'miniImageNet_category_split_train_phase_%s.pickle'           
        else:
            self.file_pattern = 'miniImageNet_category_split_%s.pickle'           
        self.data = {}
        with open(os.path.join(self.data_root, self.file_pattern % partition), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            imgs = data['data']
            self.labels = data['labels']
            self.img_label_pairs = list(zip(imgs, self.labels))
            random.shuffle(self.img_label_pairs)

    def __getitem__(self, item):
        img = np.asarray(self.img_label_pairs[item][0]).astype('uint8')
        img = self.transform(img)
        target = self.img_label_pairs[item][1] - min(self.labels)

        return img, target, "0-%s-%s.jpeg" % (target, item)
        
    def __len__(self):
        return len(self.img_label_pairs)

class TieredImageNet(Dataset):
    def __init__(self, data_root, data_aug=False, partition='train', pretrain=True, transform=None):
        super(Dataset, self).__init__()
        self.data_root = data_root
        self.partition = partition
        self.data_aug = data_aug
        self.mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        self.std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]

        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.pretrain = pretrain

        if transform is None:
            if self.partition == 'train' and self.data_aug:
                self.transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.RandomCrop(84, padding=8),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
            else:
                self.transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.transform = transform

        if self.pretrain:
            self.image_file_pattern = '%s_images.npz'
            self.label_file_pattern = '%s_labels.pkl'
        else:
            self.image_file_pattern = '%s_images.npz'
            self.label_file_pattern = '%s_labels.pkl'

        self.data = {}

        # modified code to load tieredImageNet
        image_file = os.path.join(self.data_root, self.image_file_pattern % partition)
        self.imgs = np.load(image_file)['images']
        # print(self.imgs)
        label_file = os.path.join(self.data_root, self.label_file_pattern % partition)
        self.labels = self._load_labels(label_file)['labels']
        self.img_label_pairs = list(zip(self.imgs, self.labels))
        random.shuffle(self.img_label_pairs)

    def __getitem__(self, item):
        img = np.asarray(self.img_label_pairs[item][0]).astype('uint8')
        img = self.transform(img)
        target = self.img_label_pairs[item][1] - min(self.labels)

        return img, target, "0-%s-%s.jpeg" % (target, item)

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def _load_labels(file):
        try:
            with open(file, 'rb') as fo:
                data = pickle.load(fo)
            return data
        except:
            with open(file, 'rb') as f:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                data = u.load()
            return data


class DataConfig(object):
    """Common configuration options for creating data processing pipelines."""

    def __init__(self, args):
        """Initialize a DataConfig.
        """
        # General info
        self.path: Path = Path(args.meta_dataset_path)
        self.train_batch_size: int = args.train_batch_size
        self.eval_batch_size: int = args.eval_batch_size
        # self.num_workers: int = args.num_workers
        self.shuffle: bool = True

        # Transforms and augmentations
        self.image_size = args.img_size
        self.test_transforms = args.test_transforms
        self.train_transforms = args.train_transforms

class OmniglotSampler(sampler.Sampler):
    """Samples task specification keys for an OmniglotDataset."""

    def __init__(self, split_idxs, num_training_examples, character_folders, shuffle: bool=False):
        """Inits OmniglotSampler.

        Args:
            split_idxs (range): indices that comprise the
                training/validation/test split
            batch_size (int): number of images to sample from Omniglot across characters
            shuffle (bool): whether to draw randomly or not. Set to True for training and False otherwise.
        """
        super().__init__(None)
        self._split_idxs = split_idxs
        self._shuffle = shuffle
        self._character_folders = character_folders
        self._num_training_examples = num_training_examples

    def __iter__(self):
        if not self._shuffle:
            for class_idx in self._split_idxs:
                all_file_paths = glob.glob(
                    os.path.join(self._character_folders[class_idx], '*.png')
                )
                for file_path_index in range(len(all_file_paths)):
                    yield class_idx, file_path_index
        else:
            for idx_idx in range(self._num_training_examples):
                class_idx = np.random.default_rng().choice(
                                self._split_idxs
                )
                all_file_paths = glob.glob(
                    os.path.join(self._character_folders[class_idx], '*.png')
                )
                file_path_index = np.random.default_rng().choice(range(len(all_file_paths)))
                yield class_idx, file_path_index

    def __len__(self):
        return self._num_training_examples

class DatasetSpecification(
        collections.namedtuple('DatasetSpecification',
                               ['name',
                                'classes_per_split',
                                'images_per_class',
                                'class_names',
                                'path',
                                'file_pattern'])):
    """The specification of a dataset.
        Args:
            name: string, the name of the dataset.
            classes_per_split: a dict specifying the number of classes allocated to
                each split.
            images_per_class: a dict mapping each class id to its number of images.
                Usually, the number of images is an integer, but if the dataset has
                'train' and 'test' example-level splits (or "pools"), then it is a dict
                mapping a string (the pool) to an integer indicating how many examples
                are in that pool. E.g., the number of images could be {'train': 5923,
                'test': 980}.
            class_names: a dict mapping each class id to the corresponding class name.
            path: the path to the dataset's files.
            file_pattern: a string representing the naming pattern for each class's
                file. This string should be either '{}.tfrecords' or '{}_{}.tfrecords'.
                The first gap will be replaced by the class id in both cases, while in
                the latter case the second gap will be replaced with by a shard index,
                or one of 'train', 'valid' or 'test'. This offers support for multiple
                shards of a class' images if a class is too large, that will be merged
                later into a big pool for sampling, as well as different splits that
                will be treated as disjoint pools for sampling the support versus query
                examples of an episode.
    """

    def initialize(self,
                   restricted_classes_per_split: Dict[Split, int] = None):
        """Initializes a DatasetSpecification.
        Args:
            restricted_classes_per_split: A dict that specifies for each split, a
                number to restrict its classes to. This number must be no greater than
                the total number of classes of that split. By default this is None and
                no restrictions are applied (all classes are used).
        Raises:
            ValueError: Invalid file_pattern provided.
        """
        # Check that the file_pattern adheres to one of the allowable forms
        if self.file_pattern not in ['{}.tfrecords', '{}_{}.tfrecords']:
            raise ValueError('file_pattern must be either "{}.tfrecords" or '
                             '"{}_{}.tfrecords" to support shards or splits.')

        if restricted_classes_per_split is not None:
            _check_validity_of_restricted_classes_per_split(
                restricted_classes_per_split, self.classes_per_split)

            # Apply the restriction.
            for split, restricted_num_classes in restricted_classes_per_split.items():
                self.classes_per_split[split] = restricted_num_classes

    def get_total_images_per_class(self,
                                   class_id: int = None):
        """Returns the total number of images for the specified class.
        Args:
            class_id: The class whose number of images will be returned. If this is
                None, it is assumed that the dataset has the same number of images for
                each class.
            pool: A string ('train' or 'test', optional) indicating which
                example-level split to select, if the current dataset has them.
        Raises:
            ValueError: when
                - no class_id specified and yet there is class imbalance, or
                - no pool specified when there are example-level splits, or
                - pool is specified but there are no example-level splits, or
                - incorrect value for pool.
            RuntimeError: the DatasetSpecification is out of date (missing info).
        """
        return get_total_images_per_class(self, class_id)

    def get_classes(self,
                    split: Split):
        """Gets the sequence of class labels for a split.
        Labels are returned ordered and without gaps.
        Args:
            split: A Split, the split for which to get classes.
        Returns:
            The sequence of classes for the split.
        Raises:
            ValueError: An invalid split was specified.
        """
        return get_classes(split, self.classes_per_split)

    def to_dict(self,
                ret_Dict):
        """Returns a dictionary for serialization to JSON.
        Each member is converted to an elementary type that can be serialized to
        JSON readily.
        """

        # Start with the dict representation of the namedtuple
        ret_dict = self._asdict()

        # Add the class name for reconstruction when deserialized
        ret_Dict['__class__'] = self.__class__.__name__

        # Convert Split enum instances to their name (string)
        ret_Dict['classes_per_split'] = {
            split.name: count
            for split, count in six.iteritems(ret_Dict['classes_per_split'])
        }

        # Convert binary class names to unicode strings if necessary
        class_names = {}
        for class_id, name in six.iteritems(ret_Dict['class_names']):
            if isinstance(name, six.binary_type):
                name = name.decode()
            elif isinstance(name, np.integer):
                name = six.text_type(name)

            class_names[class_id] = name
        ret_Dict['class_names'] = class_names

        return ret_dict


class BiLevelDatasetSpecification(
        collections.namedtuple('BiLevelDatasetSpecification',
                               ['name',
                                'superclasses_per_split',
                                'classes_per_superclass',
                                'images_per_class',
                                'superclass_names',
                                'class_names',
                                'path',
                                'file_pattern'])):
    """The specification of a dataset that has a two-level hierarchy.
        Args:
            name: string, the name of the dataset.
            superclasses_per_split: a dict specifying the number of superclasses
                allocated to each split.
            classes_per_superclass: a dict specifying the number of classes in each
                superclass.
            images_per_class: a dict mapping each class id to its number of images.
            superclass_names: a dict mapping each superclass id to its name.
            class_names: a dict mapping each class id to the corresponding class name.
            path: the path to the dataset's files.
            file_pattern: a string representing the naming pattern for each class's
                file. This string should be either '{}.tfrecords' or '{}_{}.tfrecords'.
                The first gap will be replaced by the class id in both cases, while in
                the latter case the second gap will be replaced with by a shard index,
                or one of 'train', 'valid' or 'test'. This offers support for multiple
                shards of a class' images if a class is too large, that will be merged
                later into a big pool for sampling, as well as different splits that
                will be treated as disjoint pools for sampling the support versus query
                examples of an episode.
    """

    def initialize(self,
                   restricted_classes_per_split: Union[Split, int] = None):
        """Initializes a DatasetSpecification.
        Args:
            restricted_classes_per_split: A dict that specifies for each split, a
                number to restrict its classes to. This number must be no greater than
                the total number of classes of that split. By default this is None and
                no restrictions are applied (all classes are used).
        Raises:
            ValueError: Invalid file_pattern provided
        """
        # Check that the file_pattern adheres to one of the allowable forms
        if self.file_pattern not in ['{}.tfrecords', '{}_{}.tfrecords']:
            raise ValueError('file_pattern must be either "{}.tfrecords" or '
                             '"{}_{}.tfrecords" to support shards or splits.')

        if restricted_classes_per_split is not None:
            # Create a dict like classes_per_split of DatasetSpecification.
            classes_per_split = {}
            for split in self.superclasses_per_split.keys():
                num_split_classes = self._count_classes_in_superclasses(
                    self.get_superclasses(split))
                classes_per_split[split] = num_split_classes

            _check_validity_of_restricted_classes_per_split(
                restricted_classes_per_split, classes_per_split)

        # The restriction in this case is applied in get_classes() below.
        self.restricted_classes_per_split = restricted_classes_per_split

    def get_total_images_per_class(self,
                                   class_id: int = None):
        """Returns the total number of images for the specified class.
        Args:
            class_id: The class whose number of images will be returned. If this is
                None, it is assumed that the dataset has the same number of images for
                each class.
            pool: A string ('train' or 'test', optional) indicating which
                example-level split to select, if the current dataset has them.
        Raises:
            ValueError: when
                - no class_id specified and yet there is class imbalance, or
                - no pool specified when there are example-level splits, or
                - pool is specified but there are no example-level splits, or
                - incorrect value for pool.
            RuntimeError: the DatasetSpecification is out of date (missing info).
        """
        return get_total_images_per_class(self, class_id)

    def get_superclasses(self, split: Split):
        """Gets the sequence of superclass labels for a split.
        Labels are returned ordered and without gaps.
        Args:
            split: A Split, the split for which to get the superclasses.
        Returns:
            The sequence of superclasses for the split.
        Raises:
            ValueError: An invalid split was specified.
        """
        return get_classes(split, self.superclasses_per_split)

    def _count_classes_in_superclasses(self, superclass_ids: List[int]):
        return sum(self.classes_per_superclass[superclass_id]
                   for superclass_id in superclass_ids)

    def _get_split_offset(self, split: Split):
        """Returns the starting class id of the contiguous chunk of ids of split.
        Args:
            split: A Split, the split for which to get classes.
        Raises:
            ValueError: Invalid dataset split.
        """
        if split == Split.TRAIN:
            offset = 0
        elif split == Split.VALID:
            previous_superclasses = range(0,
                                          self.superclasses_per_split[Split.TRAIN])
            offset = self._count_classes_in_superclasses(previous_superclasses)
        elif split == Split.TEST:
            previous_superclasses = range(0,
                                          self.superclasses_per_split[Split.TRAIN] +
                                              self.superclasses_per_split[Split.VALID])
            offset = self._count_classes_in_superclasses(previous_superclasses)
        else:
            raise ValueError('Invalid dataset split.')
        return offset

    def get_classes(self, split: Split):
        """Gets the sequence of class labels for a split.
        Labels are returned ordered and without gaps.
        Args:
            split: A Split, the split for which to get classes.
        Returns:
            The sequence of classes for the split.
        """
        if not hasattr(self, 'restricted_classes_per_split'):
            self.initialize()

        offset = self._get_split_offset(split)

        if (self.restricted_classes_per_split is not None and
                split in self.restricted_classes_per_split):
            num_split_classes = self.restricted_classes_per_split[split]
        else:
            # No restriction, so include all classes of the given split.
            num_split_classes = self._count_classes_in_superclasses(
                self.get_superclasses(split))

        return range(offset, offset + num_split_classes)

    def get_class_ids_from_superclass_subclass_inds(self,
                                                    split: Split,
                                                    superclass_id: int,
                                                    class_inds: List[int]):
        """Gets the class ids of a number of classes of a given superclass.
        Args:
            split: A Split, the split for which to get classes.
            superclass_id: An int. The id of a superclass.
            class_inds: A list or sequence of ints. The indices into the classes of
                the superclass superclass_id that we wish to return class id's for.
        Returns:
            rel_class_ids: A list of ints of length equal to that of class_inds. The
                class id's relative to the split (between 0 and num classes in split).
            class_ids: A list of ints of length equal to that of class_inds. The class
                id's relative to the dataset (between 0 and the total num classes).
        """
        # The number of classes before the start of superclass_id, i.e. the class id
        # of the first class of the given superclass.
        superclass_offset = self._count_classes_in_superclasses(range(superclass_id))

        # Absolute class ids (between 0 and the total number of dataset classes).
        class_ids = [superclass_offset + class_ind
                     for class_ind in class_inds]

        # Relative (between 0 and the total number of classes in the split).
        # This makes the assumption that the class id's are in a contiguous range.
        rel_class_ids = [class_id - self._get_split_offset(split)
                         for class_id in class_ids]

        return rel_class_ids, class_ids

    def to_dict(self, ret_Dict):
        """Returns a dictionary for serialization to JSON.
        Each member is converted to an elementary type that can be serialized to
        JSON readily.
        """
        # Start with the dict representation of the namedtuple
        ret_dict = self._asdict()

        # Add the class name for reconstruction when deserialized
        ret_Dict['__class__'] = self.__class__.__name__

        # Convert Split enum instances to their name (string)
        ret_Dict['superclasses_per_split'] = {
            split.name: count
            for split, count in six.iteritems(ret_Dict['superclasses_per_split'])
        }

        return ret_dict


class TFRecordDataset(IterableDataset):
    """Parse (generic) TFRecords dataset into `IterableDataset` object,
    which contain `np.ndarrays`s. By default (when `sequence_description`
    is None), it treats the TFRecords as containing `tf.Example`.
    Otherwise, it assumes it is a `tf.SequenceExample`.
    Params:
    -------
    data_path: str
        The path to the tfrecords file.
    index_path: str or None
        The path to the index file.
    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.
    shuffle: int, optional, default=None
        Length of buffer. Determines how many records are queued to
        sample from.
    transform : a callable, default = None
        A function that takes in the input `features` i.e the dict
        provided in the description, transforms it and returns a
        desirable output.
    sequence_description: list or dict of str, optional, default=None
        Similar to `description`, but refers to the sequence features
        within a `SequenceExample`. When this field is `None`, then it
        is assumed that an `Example` is being read otherwise, a
        `SequenceExample` is read. If an empty list or dictionary is
        passed, then all features contained in the file are extracted.
    """

    def __init__(self,
                 data_path: str,
                 index_path,
                 description = None,
                 shuffle = None,
                 sequence_description = None,
                 ) -> None:
        super(TFRecordDataset, self).__init__()
        self.data_path = data_path
        self.index_path = index_path
        self.description = description
        self.sequence_description = sequence_description
        self.shuffle = shuffle
        self.random_gen = np.random.RandomState()

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            shard = worker_info.id, worker_info.num_workers
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
        else:
            shard = None
        it = tfrecord_loader(data_path=self.data_path,
                                    index_path=self.index_path,
                                    description=self.description,
                                    shard=shard,
                                    shuffle=self.shuffle,
                                    sequence_description=self.sequence_description,
                                    random_gen=self.random_gen)
        return it


class Reader(object):
    """Class reading data from one source and assembling examples.
    Specifically, it holds part of a tf.data pipeline (the source-specific part),
    that reads data from TFRecords and assembles examples from them.
    """

    def __init__(self,
                 dataset_spec: Union['BiLevelDatasetSpecification', 'DatasetSpecification'],
                 split: Split,
                 shuffle: bool,
                 offset: int):
        """Initializes a Reader from a source.
        The source is identified by dataset_spec and split.
        Args:
          dataset_spec: DatasetSpecification, dataset specification.
          split: A learning_spec.Split object identifying the source split.
          shuffle_buffer_size: An integer, the shuffle buffer size for each Dataset
            object. If 0, no shuffling operation will happen.
          read_buffer_size_bytes: int or None, buffer size for each TFRecordDataset.
          num_to_take: Optional, an int specifying a number of elements to pick from
            each tfrecord. If specified, the available images of each class will be
            restricted to that int. By default (-1) no restriction is applied and
            all data is used.
        """
        self.split = split
        self.dataset_spec = dataset_spec
        self.offset = offset
        self.shuffle = shuffle

        self.base_path = self.dataset_spec.path
        self.class_set = self.dataset_spec.get_classes(self.split)
        self.num_classes = len(self.class_set)

    def construct_class_datasets(self):
        """Constructs the list of class datasets.
        Returns:
          class_datasets: list of tf.data.Dataset, one for each class.
        """
        record_file_pattern = self.dataset_spec.file_pattern
        index_file_pattern = '{}.index'
        # We construct one dataset object per class. Each dataset outputs a stream
        # of `(example_string, dataset_id)` tuples.
        class_datasets = []
        for dataset_id in range(self.num_classes):
            class_id = self.class_set[dataset_id]  # noqa: E111
            if record_file_pattern.startswith('{}_{}'):
                # TODO(lamblinp): Add support for sharded files if needed.
                raise NotImplementedError('Sharded files are not supported yet. '  # noqa: E111
                                          'The code expects one dataset per class.')
            elif record_file_pattern.startswith('{}'):
                data_path = os.path.join(self.base_path, record_file_pattern.format(class_id))  # noqa: E111
                index_path = os.path.join(self.base_path, index_file_pattern.format(class_id))  # noqa: E111
            else:
                raise ValueError('Unsupported record_file_pattern in DatasetSpec: %s. '  # noqa: E111
                                 'Expected something starting with "{}" or "{}_{}".' %
                                 record_file_pattern)
            description = {"image": "byte", "label": "int"}

            # decode_fn = partial(self.decode_image, offset=self.offset)
            dataset = TFRecordDataset(data_path=data_path,
                                      index_path=index_path,
                                      description=description,
                                      shuffle=self.shuffle)

            class_datasets.append(dataset)

        assert len(class_datasets) == self.num_classes
        return class_datasets


class BatchDataset(IterableDataset):
    def __init__(self,
                 class_datasets,
                 transforms = None):
        super().__init__()

        self.class_datasets = class_datasets
        self.transforms = transforms
        self.random_gen = np.random.RandomState()

    def __iter__(self):
        while True:
            rand_class = self.random_gen.randint(len(self.class_datasets))
            sample_dic = self.get_next(rand_class)
            sample_dic = parse_record(sample_dic)
            transformed_image = self.transforms(sample_dic['image']) if self.transforms else sample_dic['image']
            target = sample_dic['label'][0]
            yield transformed_image, target, "0-%s-%s.jpeg" % (target, sample_dic['id'])

    def get_next(self, class_id):
        try:
            sample_dic = next(self.class_datasets[class_id])
        except (StopIteration, TypeError) as e:
            self.class_datasets[class_id] = cycle_(self.class_datasets[class_id])
            sample_dic = next(self.class_datasets[class_id])

        return sample_dic


class ZipDataset(IterableDataset):
    def __init__(self, dataset_list):
        self.dataset_list = dataset_list
        self.random_gen = np.random.RandomState()

    def __iter__(self):
        while True:
            rand_source = self.random_gen.randint(len(self.dataset_list))
            next_e = self.get_next(rand_source)

            yield next_e

    def get_next(self, source_id):
        try:
            dataset = next(self.dataset_list[source_id])
        except (StopIteration, TypeError) as e:
            self.dataset_list[source_id] = iter(self.dataset_list[source_id])
            dataset = next(self.dataset_list[source_id])

        return dataset


def as_dataset_spec(dct: Dict[str, Any]):
    """Hook to `json.loads` that builds a DatasetSpecification from a dict.
    Args:
         dct: A dictionary with string keys, corresponding to a JSON file.
    Returns:
        Depending on the '__class__' key of the dictionary, a DatasetSpecification,
        or BiLevelDatasetSpecification. Defaults
        to returning `dct`.
    """
    def _key_to_int(dct):
        """Returns a new dictionary whith keys converted to ints."""
        return {int(key): value
                for key, value in six.iteritems(dct)}

    def _key_to_split(dct):
        """Returns a new dictionary whith keys converted to Split enums."""
        return {Split[key]: value
                for key, value in six.iteritems(dct)}

    # Early returns:
    if '__class__' not in dct:
        return dct
    if dct['__class__'] not in ['DatasetSpecification',
                                'BiLevelDatasetSpecification']:
        return dct

    # Actual datasets
    if dct['__class__'] == 'DatasetSpecification':
        images_per_class = {}
        for class_id, n_images in six.iteritems(dct['images_per_class']):
            # If n_images is a dict, it maps each class ID to a string->int
            # dictionary containing the size of each pool.
            if isinstance(n_images, dict):
                # Convert the number of classes in each pool to int.
                n_images = {pool: int(pool_size)
                            for pool, pool_size in six.iteritems(n_images)}
            else:
                n_images = int(n_images)
            images_per_class[int(class_id)] = n_images

        return DatasetSpecification(
            name=dct['name'],
            classes_per_split=_key_to_split(dct['classes_per_split']),
            images_per_class=images_per_class,
            class_names=_key_to_int(dct['class_names']),
            path=dct['path'],
            file_pattern=dct['file_pattern'])

    elif dct['__class__'] == 'BiLevelDatasetSpecification':
        return BiLevelDatasetSpecification(
            name=dct['name'],
            superclasses_per_split=_key_to_split(dct['superclasses_per_split']),
            classes_per_superclass=_key_to_int(dct['classes_per_superclass']),
            images_per_class=_key_to_int(dct['images_per_class']),
            superclass_names=_key_to_int(dct['superclass_names']),
            class_names=_key_to_int(dct['class_names']),
            path=dct['path'],
            file_pattern=dct['file_pattern'])

    else:  # Should not arrive there, as it's covered by the early return
        raise ValueError(dct.__repr__())


def load_dataset_spec(dataset_records_path: str, convert_from_pkl: bool = False):
    """Loads dataset specification from directory containing the dataset records.
    Newly-generated datasets have the dataset specification serialized as JSON,
    older ones have it as a .pkl file. If no JSON file is present and
    `convert_from_pkl` is passed, this method will load the .pkl and serialize it
    to JSON.
    Args:
        dataset_records_path: A string, the path to the directory containing
            .tfrecords files and dataset_spec.
        convert_from_pkl: A boolean (False by default), whether to convert a
            dataset_spec.pkl file to JSON.
    Returns:
        A DatasetSpecification, BiLevelDatasetSpecification, or depending on the dataset.
    Raises:
        RuntimeError: If no suitable dataset_spec file is found in directory
            (.json or .pkl depending on `convert_from_pkl`).
    """
    json_path = os.path.join(dataset_records_path, 'dataset_spec.json')
    pkl_path = os.path.join(dataset_records_path, 'dataset_spec.pkl')

    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data_spec = json.load(f, object_hook=as_dataset_spec)
    elif os.path.exists(pkl_path):
        if convert_from_pkl:
            logging.info('Loading older dataset_spec.pkl to convert it.')
            with open(pkl_path, 'rb') as f:
                data_spec = pkl.load(f)
            with open(json_path, 'w') as f:
                json.dump(data_spec.to_dict(), f, indent=2)
        else:
            raise RuntimeError(
                'No dataset_spec.json file found in directory %s, but an older '
                'dataset_spec.pkl was found. You can try to pass '
                '`convert_from_pkl=True` to convert it, or you may need to run the '
                'conversion again in order to make sure you have the latest version.'
                % dataset_records_path)
    else:
        raise RuntimeError('No dataset_spec file found in directory %s' % dataset_records_path)

    # Replace outdated path of where to find the dataset's records.
    data_spec = data_spec._replace(path=dataset_records_path)
    return data_spec


def get_dataspecs(args, source: str):
    # Recovering data
    data_config = DataConfig(args=args)

    dataset_records_path = data_config.path / source
    # Original codes handles paths as strings:
    dataset_spec = load_dataset_spec(str(dataset_records_path))

    return dataset_spec, data_config

def make_batch_pipeline(dataset_spec,
                        data_config,
                        split) -> Dataset:
    """Returns a pipeline emitting data from potentially multiples source as batches.
    Args:
      dataset_spec: A list of DatasetSpecification object defining what to read from.
      split: A learning_spec.Split object identifying the source (meta-)split.
    Returns:
    """

    offset = 0
    dataset_list = []
    batch_reader = Reader(dataset_spec=dataset_spec,
                          split=split,
                          shuffle=data_config.shuffle,
                          offset=offset)

    class_datasets = batch_reader.construct_class_datasets()

    transforms = get_transforms(data_config=data_config, split=split)
    dataset = BatchDataset(class_datasets=class_datasets,
                           transforms=transforms)
    dataset_list.append(dataset)

    return ZipDataset(dataset_list)

# only needed for Iterable-style datasets
def worker_init_fn_(worker_id, seed):
    worker_info = get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    random_gen = np.random.RandomState(seed + worker_id)

    dataset.random_gen = random_gen
    for source_dataset in dataset.dataset_list:
        source_dataset.random_gen = random_gen

        for class_dataset in source_dataset.class_datasets:
            class_dataset.random_gen = random_gen


def get_loader(args, test_on_train=False):
    # train_split_idxs = range(NUM_TRAIN_CLASSES)
    # val_split_idxs = range(
    #     NUM_TRAIN_CLASSES,
    #     NUM_TRAIN_CLASSES + NUM_VAL_CLASSES
    # )
    # test_split_idxs = range(
    #     NUM_TRAIN_CLASSES + NUM_VAL_CLASSES,
    #     NUM_TRAIN_CLASSES + NUM_VAL_CLASSES + NUM_TEST_CLASSES
    # )

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # transform_train = transforms.Compose([
    #     transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # ])
    # transform_test = transforms.Compose([
    #     transforms.Resize((args.img_size, args.img_size)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # ])


    if args.dataset == 'omniglot':
        # if necessary, download the Omniglot dataset
        if not os.path.isdir(_OMNIGLOT_BASE_PATH):
            gdd.GoogleDriveDownloader.download_file_from_google_drive(
                file_id=_OMNIGLOT_GDD_FILE_ID,
                dest_path=f'{_OMNIGLOT_BASE_PATH}.zip',
                unzip=True
            )
        # split character folders into train/valid/test
        character_folders = glob.glob(os.path.join(_OMNIGLOT_BASE_PATH, '*/*/'))
        np.random.default_rng(0).shuffle(character_folders)
        trainset = OmniglotDataset(character_folders, partition='train')
        validset = OmniglotDataset(character_folders, partition='val')
        testset = OmniglotDataset(character_folders, partition='test') if args.local_rank in [-1, 0] else None
        train_sampler = OmniglotSampler(range(OMNIGLOT_NUM_TRAIN_CLASSES), args.num_steps, character_folders, not test_on_train)
        valid_sampler = OmniglotSampler(range(
                OMNIGLOT_NUM_TRAIN_CLASSES,
                OMNIGLOT_NUM_TRAIN_CLASSES + OMNIGLOT_NUM_VAL_CLASSES
            ), args.num_steps, character_folders, False)
        test_sampler = OmniglotSampler(range(
                OMNIGLOT_NUM_TRAIN_CLASSES + OMNIGLOT_NUM_VAL_CLASSES,
                OMNIGLOT_NUM_TRAIN_CLASSES + OMNIGLOT_NUM_VAL_CLASSES + OMNIGLOT_NUM_TEST_CLASSES
            ), args.num_steps, character_folders, False)
        worker_init_fn = None
    elif args.dataset == 'miniimagenet':
        trainset = MiniImageNet(_MINIIMAGENET_BASE_PATH, args.augment_mini_imagenet, partition='train', pretrain=True, transform=None)
        validset = MiniImageNet(_MINIIMAGENET_BASE_PATH, args.augment_mini_imagenet, partition='val', pretrain=True, transform=None)
        testset = MiniImageNet(_MINIIMAGENET_BASE_PATH, args.augment_mini_imagenet, partition='test', pretrain=True, transform=None)
        train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
        valid_sampler = SequentialSampler(validset)
        test_sampler = SequentialSampler(testset)
        worker_init_fn = None
    elif args.dataset == 'tieredimagenet':
        trainset = TieredImageNet(_MINIIMAGENET_BASE_PATH, args.augment_mini_imagenet, partition='train', pretrain=True, transform=None)
        validset = TieredImageNet(_MINIIMAGENET_BASE_PATH, args.augment_mini_imagenet, partition='val', pretrain=True, transform=None)
        testset = TieredImageNet(_MINIIMAGENET_BASE_PATH, args.augment_mini_imagenet, partition='test', pretrain=True, transform=None)
        train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
        valid_sampler = SequentialSampler(validset)
        test_sampler = SequentialSampler(testset)
        worker_init_fn = None
    elif args.dataset in set(["aircraft", "cu_birds", "dtd", "fungi", "mscoco", "traffic_sign", "vgg_flower"]):
        dataset_spec, data_config = get_dataspecs(args, args.dataset)
        trainset = make_batch_pipeline(dataset_spec=dataset_spec,
                              data_config=data_config,
                              split=Split.TRAIN)
        validset = make_batch_pipeline(dataset_spec=dataset_spec,
                              data_config=data_config,
                              split=Split.VALID)
        testset = make_batch_pipeline(dataset_spec=dataset_spec,
                              data_config=data_config,
                              split=Split.TEST)
        # iterable-style datasets, so no sampler allowed to be specified in DataLoader
        train_sampler = None
        valid_sampler = None
        test_sampler = None

        worker_init_fn = functools.partial(worker_init_fn_, seed=args.seed)

    if args.local_rank == 0:
        torch.distributed.barrier()
    # train_sampler = OmniglotSampler(train_split_idxs, args.train_batch_size)
    # test_sampler = OmniglotSampler(test_split_idxs, args.eval_batch_size)
    

    train_loader = DataLoader(
        dataset=trainset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn
    )

    valid_loader = DataLoader(
        dataset=validset,
        batch_size=args.eval_batch_size,
        sampler=valid_sampler,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn
    ) if validset is not None else None

    test_loader = DataLoader(
        dataset=testset,
        batch_size=args.eval_batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn
    ) if testset is not None else None

    return train_loader, valid_loader, test_loader
