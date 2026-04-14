""" # data/dataset.py
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.datasets as tv_datasets


class FewShotDataset(Dataset):
    def __init__(self, root: str, dataset_name: str = "cifar100", split: str = 'train', transform=None):
        self.root = root
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.transform = transform
        self.label_names = None

        if self.dataset_name in ["cifar_fs", "cifar100"]:
            self._load_cifar100()
        elif self.dataset_name == "cub":
            self._load_cub()
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def _load_cifar100(self):
        is_train = (self.split == "train")
        cifar = tv_datasets.CIFAR100(root=self.root, train=is_train, download=True, transform=None)
        self.images = [Image.fromarray(img) for img in cifar.data]
        self.labels = cifar.targets
        self.label_names = cifar.classes

    def _load_cub(self):
        cub_root = os.path.join(self.root, "CUB_200_2011")
        if not os.path.exists(cub_root):
            raise FileNotFoundError(f"CUB-200-2011 not found at {cub_root}")

        # Check for standard files
        images_txt = os.path.join(cub_root, "images.txt")
        labels_txt = os.path.join(cub_root, "image_class_labels.txt")
        split_txt = os.path.join(cub_root, "train_test_split.txt")

        if not os.path.exists(images_txt):
            raise FileNotFoundError(f"Missing {images_txt}. Please download full CUB-200-2011 dataset.")

        # Load data
        with open(images_txt, 'r') as f:
            img_paths = [line.strip().split()[1] for line in f.readlines()]
        with open(labels_txt, 'r') as f:
            img_labels = [int(line.strip().split()[1]) - 1 for line in f.readlines()]
        with open(split_txt, 'r') as f:
            splits = [int(line.strip().split()[1]) for line in f.readlines()]

        # Filter by split
        split_val = 1 if self.split == "train" else 0
        keep_idx = [i for i, s in enumerate(splits) if s == split_val]

        self.images = []
        self.labels = []
        for i in keep_idx:
            img_path = os.path.join(cub_root, "images", img_paths[i])
            if os.path.exists(img_path):
                self.images.append(Image.open(img_path).convert('RGB'))
                self.labels.append(img_labels[i])

        # Load real class names
        classes_txt = os.path.join(cub_root, "classes.txt")
        with open(classes_txt, 'r') as f:
            self.label_names = [line.strip().split(".")[1].replace("_", " ") for line in f.readlines()]

        print(f"Loaded CUB-200 {self.split} split: {len(self.images)} images, {len(self.label_names)} classes")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


# ====================== Episodic Sampler ======================
def episodic_sampler(dataset: FewShotDataset, n_way: int, k_shot: int, q_query: int):
    all_labels = np.array(dataset.labels)
    unique_classes = np.unique(all_labels)
    
    selected_classes = np.random.choice(unique_classes, n_way, replace=False)

    support_imgs, query_imgs = [], []
    support_labels, query_labels = [], []
    class_names = []

    for idx, cls in enumerate(selected_classes):
        class_names.append(dataset.label_names[cls])
        
        indices = np.where(all_labels == cls)[0]
        np.random.shuffle(indices)

        for i in indices[:k_shot]:
            img, _ = dataset[i]
            support_imgs.append(img)
            support_labels.append(idx)

        for i in indices[k_shot : k_shot + q_query]:
            img, _ = dataset[i]
            query_imgs.append(img)
            query_labels.append(idx)

    support_imgs = torch.stack(support_imgs)
    query_imgs = torch.stack(query_imgs)
    support_labels = torch.tensor(support_labels)
    query_labels = torch.tensor(query_labels)

    return support_imgs, query_imgs, support_labels, class_names, query_labels


# ====================== Transforms ======================
def get_transforms(dataset_name: str = "cifar_fs", image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ]) """


# data/dataset.py
import os
import pickle
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class FewShotDataset(Dataset):
    def __init__(self, root: str, dataset_name: str = "miniImageNet", split: str = 'test', transform=None):
        # Set root to the folder CONTAINING CUB_200_2011
        self.root = '/workdir1.8t/fei27/CGT/APDFA/data/miniImageNet'
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_names = []

        self._load_miniImageNet()

    


    def _load_miniImageNet(self):
        """
        Load tiered-ImageNet from folder structure (synsets).
        Path: root/tiered_imagenet/[train|val|test]/[synset_id]/*.jpg
        """
        # Based on your error, the split folder is at:
        split_dir = os.path.join(self.root, "miniImageNet", self.split)
        
        if not os.path.exists(split_dir):
            # Try one level deeper just in case
            split_dir = os.path.join(self.root, "/workdir1.8t/fei27/CGT/APDFA/data/miniImageNet", self.split)
            if not os.path.exists(split_dir):
                raise FileNotFoundError(f"Could not find split directory at {split_dir}")

        print(f"⌛ Scanning folders in {split_dir}...")

        # Get all class folders (synsets)
        self.class_folders = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        
        self.images = []
        self.labels = []
        self.label_names = []

        for idx, synset in enumerate(self.class_folders):
            class_path = os.path.join(split_dir, synset)
            
            # Store synset as the label name (e.g., n03447447)
            self.label_names.append(synset)
            
            # Find all images in this class folder
            class_images = [os.path.join(class_path, f) for f in os.listdir(class_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_path in class_images:
                self.images.append(img_path)
                self.labels.append(idx)

        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {split_dir}")

        print(f"✅ Loaded {len(self.images)} images from {len(self.class_folders)} classes for {self.split}.")

    def __getitem__(self, idx):
        # 1. Get the path
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 2. Load the image from disk
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"CRITICAL: Failed to load {img_path}. Error: {e}")
            img = Image.new('RGB', (224, 224), color=0)  # fallback
        
        # 3. Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, label
# ====================== Fully Corrected Sampler ======================
def episodic_sampler(dataset, n_way: int, k_shot: int, q_query: int):
    all_labels = np.array(dataset.labels)
    unique_classes = np.unique(all_labels)
    
    # Randomly select N classes
    selected_classes = np.random.choice(unique_classes, n_way, replace=False)

    support_imgs, query_imgs = [], []
    support_labels, query_labels = [], []
    class_names = []

    for idx, cls in enumerate(selected_classes):
        class_names.append(dataset.label_names[cls])
        
        # Get all indices for this class
        indices = np.where(all_labels == cls)[0]
        np.random.shuffle(indices)

        # Handle support images
        if k_shot > 0:
            for i in indices[:k_shot]:
                img, _ = dataset[i]
                support_imgs.append(img)
                support_labels.append(idx)

        # Handle query images
        for i in indices[k_shot : k_shot + q_query]:
            img, _ = dataset[i]
            query_imgs.append(img)
            query_labels.append(idx)

    # Convert to Tensors (Handle 0-Shot correctly)
    if k_shot > 0:
        s_imgs = torch.stack(support_imgs)
        s_labels = torch.tensor(support_labels, dtype=torch.long)
    else:
        # Zero-Shot format
        s_imgs = torch.zeros(0, 3, 224, 224)
        s_labels = torch.tensor([], dtype=torch.long)

    # Query images MUST exist or something is wrong with dataset/indices
    if len(query_imgs) == 0:
        raise RuntimeError("Selected classes have no query images. Ensure k_shot + q_query < images per class.")
        
    q_imgs = torch.stack(query_imgs)
    q_labels = torch.tensor(query_labels, dtype=torch.long)

    return s_imgs, q_imgs, s_labels, class_names, q_labels
# ====================== Transforms ======================
def get_transforms(dataset_name: str = "miniImageNet", image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ]) 