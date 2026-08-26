"""Create and inspect contiguous Mall Dataset train/validation/test splits."""

from torch.utils.data import DataLoader, Subset

from mall_torch_dataset import MallDataset


dataset = MallDataset()

train_indices = range(0, 1400)
validation_indices = range(1400, 1700)
test_indices = range(1700, 2000)

train_dataset = Subset(dataset, train_indices)
validation_dataset = Subset(dataset, validation_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

split_total = len(train_dataset) + len(validation_dataset) + len(test_dataset)
assert split_total == len(dataset) == 2000

if __name__ == "__main__":
    print(f"Total dataset length: {len(dataset)}")
    print(f"Train length:         {len(train_dataset)}")
    print(f"Validation length:    {len(validation_dataset)}")
    print(f"Test length:          {len(test_dataset)}")
    print(f"Train source frames:      {train_indices.start + 1}-{train_indices.stop}")
    print(
        "Validation source frames: "
        f"{validation_indices.start + 1}-{validation_indices.stop}"
    )
    print(f"Test source frames:       {test_indices.start + 1}-{test_indices.stop}")
    print(f"Split total check:    {split_total}")

    for split_name, loader in (
        ("Train", train_loader),
        ("Validation", validation_loader),
        ("Test", test_loader),
    ):
        image_batch, density_batch = next(iter(loader))
        print(f"{split_name} image batch shape:   {tuple(image_batch.shape)}")
        print(f"{split_name} density batch shape: {tuple(density_batch.shape)}")
