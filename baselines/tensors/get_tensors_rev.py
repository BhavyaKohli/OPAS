import sys
sys.path.append("../../")
from main import *

if __name__ == "__main__":
    for dataset in ['audio_500', 'speech', 'cifar', 'lsun'][:1]:
        try:
            print(f"Getting {dataset}")
            TEST_FILE = f"../../final_data/{dataset}/dataset_test.hdf5"
            if dataset in ['cifar', 'lsun']:
                TEST_FILE = TEST_FILE.replace(".hdf5", "_orig.hdf5")
            test_dataset = PairDatasetTest(TEST_FILE)

            torch.manual_seed(6969) # reproducibility
            loader = test_dataset.get_dataloader(batch_size=10, shuffle=True)
            q, l = next(iter(loader))
            tensors = {'q': q, 'l': l, 'c': torch.from_numpy(test_dataset.c)}

            torch.save(tensors, f"tensors_{dataset}.pt")
        except Exception as e:
            print(f"Dataset {dataset} failed with error: {e}")