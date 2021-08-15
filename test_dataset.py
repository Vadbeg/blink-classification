from blink_classification.modules.data.dataset import BlinkDataset

if __name__ == '__main__':
    blink_dataset = BlinkDataset(data_folder='/home/vadbeg/Downloads/mrlEyes_2018_01')

    print(f'Blink dataset length: {len(blink_dataset)}')
