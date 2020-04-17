from datetime import datetime
import torch
from torch.utils.data import DataLoader
from utils.data.CountyDataset import CountyDataset


def main():
    print('-'*89)
    print('Dataset Processing Example')
    print('-'*89)

    master_path = 'data/master_county_state_reference.csv'
    mbatch = 128
    num_workers = 4

    if torch.cuda.is_available():
        device = torch.device('cuda')
        pin_memory = True
    else:
        device = torch.device('cpu')
        pin_memory = False

    print(f'device type {device.type} '
          f'| workers {num_workers} '
          f'| minibatch {mbatch}'
          f'| pin_memory {pin_memory}')

    dataset = CountyDataset(master_path)
    print(f'dataset samples {len(dataset):,}')

    X, Y = next(iter(dataset))
    print(f'X {X.size()} | Y {Y.size()}')

    print(f'batching data...')
    loader = DataLoader(dataset, batch_size=mbatch,
                        shuffle=True, pin_memory=pin_memory,
                        num_workers=num_workers)
    start = datetime.now()
    for i, (X, Y) in enumerate(loader):
        if i == 0:
            print(f'loading batches size X {X.size()} Y {Y.size()}')

    elapsed = (datetime.now()-start).total_seconds()
    print(f'donzo... loaded {i} batches in {elapsed:,.0f} seconds')


if __name__ == '__main__':
    main()
