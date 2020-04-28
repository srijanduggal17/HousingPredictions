from datetime import datetime
import torch
from torch.utils.data import DataLoader
from utils.data.CountyDataset import (
    CountyDataset,
)


def get_dataset(master_path):
    dataset = CountyDataset(master_path)
    X, Y = next(iter(dataset))

    print('-'*89)
    print('Total Dataset Summary')
    print('-'*89)

    print(f'total samples {len(dataset):,} '
          f'X {X.size()} | Y {Y.size()}')

    return dataset


def get_numpy(dataset):
    data = []
    for i in range(len(dataset)):
        X, Y = dataset[i]
        data.append((X.numpy(), Y.numpy()))

    print('-'*89)
    print(f'Numpy Dataset Summary')
    print('-'*89)
    X, Y = zip(*data)
    print(f'total samples {len(X)} '
          f'| input time periods {len(X[0])} '
          f'| input features {X[0].shape[1]} '
          f'| output time periods {len(Y[0])}'
          f'| target col {dataset.target_col}')

    return data, dataset.xcols, dataset.target_col, dataset.date_index


def check_batches(master_path, mbatch=128, num_workers=4):
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

    dataset = get_dataset(master_path)
    get_numpy(dataset)

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


def check_dfs(master_path):
    print('-'*89)
    print('CHECKING DATAFRAMES')
    print('-'*89)
    dataset = get_dataset(master_path)
    print(f'number of dataframes {len(dataset)}')
    df = dataset._get_df(0)
    print(df.head())
    print(df.columns.values)
    print(df)
#     for ndx in range(len(dataset)):
#         df = dataset._get_df(ndx)
#         print(df.shape)
#     print(f'sample df...')
#     print(dfs[0])


def main():
    print('-'*89)
    print('Dataset Processing Example')
    print('-'*89)

    master_path = 'data/master_county_state_reference.csv'
    get_dataset(master_path)


if __name__ == '__main__':
    main()
